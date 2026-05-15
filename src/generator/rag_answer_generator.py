from __future__ import annotations

import os
from typing import Any

try:
    from ollama import Client as OllamaClient
except ImportError:  # pragma: no cover
    OllamaClient = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


class RAGAnswerGenerator:
    """Generador de respuesta RAG basado en LLM.

    Recibe una query y una lista de documentos recuperados; construye el prompt
    y devuelve una respuesta final usando un modelo de chat.
    """

    def __init__(self) -> None:
        self.provider = self._read_env("RAG_LLM_PROVIDER", "openai").lower()
        self.model = self._read_env("RAG_LLM_MODEL", self._default_model_for_provider())
        self.base_url = self._read_env("RAG_LLM_BASE_URL", self._default_base_url_for_provider())
        self.api_key = self._read_env("RAG_LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.temperature = self._read_float_env("RAG_LLM_TEMPERATURE", 0.2)
        self.max_tokens = self._read_int_env("RAG_LLM_MAX_TOKENS", 420)
        self.timeout_seconds = self._read_int_env("RAG_LLM_TIMEOUT_SECONDS", 45)
        self.enabled = self._read_bool_env("RAG_LLM_ENABLED", default=True)
        self._init_error = ""
        self.client = None
        if self.provider == "ollama":
            if OllamaClient is None:
                self._init_error = "Falta instalar el paquete 'ollama'. Ejecuta: pip install -r requirements.txt"
                return
            try:
                self.client = OllamaClient(host=self.base_url)
            except Exception as exc:
                self._init_error = f"No se pudo inicializar Ollama: {exc}"
            return

        if self.provider == "openai":
            if OpenAI is None:
                self._init_error = "Falta instalar el paquete 'openai'. Ejecuta: pip install -r requirements.txt"
            return

        self._init_error = (
            f"Proveedor LLM no soportado: {self.provider}. "
            "Usa RAG_LLM_PROVIDER=ollama o RAG_LLM_PROVIDER=openai."
        )

    def build_prompt(self, query: str, documents: list[Any]) -> str:
        if not documents:
            context_block = "No se recuperaron documentos relevantes."
        else:
            parts: list[str] = []
            for index, doc in enumerate(documents, start=1):
                citation_id = self._get_doc_field(doc, "citation_id", index)
                title = self._to_text(self._get_doc_field(doc, "title", f"Documento {index}"), f"Documento {index}")
                url = self._to_text(self._get_doc_field(doc, "url", ""), "") or "N/D"
                score = self._to_float(self._get_doc_field(doc, "score", 0.0), 0.0)
                summary = self._to_text(self._get_doc_field(doc, "summary", ""), "")
                content_text = self._to_text(self._get_doc_field(doc, "content_text", ""), "")
                excerpt = self._best_excerpt(summary, content_text, title)

                parts.append(
                    "\n".join(
                        [
                            f"[{citation_id}] Titulo: {title}",
                            f"URL: {url}",
                            f"Score: {score:.4f}",
                            f"Contexto: {excerpt}",
                        ]
                    )
                )
            context_block = "\n\n".join(parts)

        return "\n".join(
            [
                "Eres un asistente RAG especializado en turismo.",
                "Responde unicamente con la evidencia del contexto recuperado.",
                "Reglas:",
                "1. No inventes hechos, precios, fechas o ubicaciones que no aparezcan en el contexto.",
                "2. Si la evidencia es insuficiente o parcial, dilo explicitamente.",
                "3. Integra detalles concretos del contexto y ancla cada idea con citas [1], [2], etc.",
                "4. Prioriza claridad, sintesis y fidelidad al contexto recuperado.",
                "5. Responde en espanol.",
                "",
                "Consulta del usuario:",
                query.strip(),
                "",
                "Contexto recuperado:",
                context_block,
                "",
                "Formato esperado:",
                "- Un parrafo breve que responda la consulta.",
                "- Uno o dos detalles complementarios si aportan valor.",
                "- No extrapoles mas alla de la evidencia.",
            ]
        )

    def generate(self, query: str, documents: list[Any], *, prompt: str | None = None) -> str:
        if self._init_error:
            return self._init_error

        if not self.enabled:
            return "El generador LLM de RAG esta deshabilitado (RAG_LLM_ENABLED=false)."

        if not self.model:
            return "No hay modelo configurado para el generador LLM de RAG."

        active_prompt = prompt or self.build_prompt(query, documents)
        if self.provider == "ollama":
            return self._generate_with_ollama(active_prompt)
        if self.provider == "openai":
            answer = self._generate_with_openai(active_prompt)
            if self._should_fallback_to_local(answer):
                return self._generate_local_answer(query, documents)
            return answer
        return f"Proveedor LLM no soportado: {self.provider}"

    def _generate_with_ollama(self, prompt: str) -> str:
        if self.client is None:
            return "Cliente de Ollama no inicializado."

        try:
            response = self.client.chat(  # type: ignore[union-attr]
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente RAG. Responde en espanol usando solo evidencia "
                            "del contexto. Incluye citas [n] y explicita cuando falte evidencia."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
        except Exception as exc:
            return (
                "No se pudo generar la respuesta con Ollama. "
                f"Asegura que Ollama este activo y que exista el modelo '{self.model}'. Error: {exc}"
            )

        message = response.get("message") if isinstance(response, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            cleaned = content.strip()
            if cleaned:
                return cleaned
        return "Ollama devolvio una respuesta vacia."

    def _generate_with_openai(self, prompt: str) -> str:
        is_local = self._is_local_base_url()
        if not self.api_key and not is_local:
            return "No hay API key configurada para OpenAI (RAG_LLM_API_KEY u OPENAI_API_KEY)."

        if self.client is None:
            try:
                self.client = OpenAI(
                    api_key=self.api_key if self.api_key else "localdev",
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                return f"No se pudo inicializar el cliente OpenAI: {exc}"

        try:
            response = self.client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente RAG. Responde en espanol usando solo evidencia "
                            "del contexto. Incluye citas [n] y explicita cuando falte evidencia."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            return f"No se pudo generar la respuesta con el LLM: {exc}"

        if not response.choices:
            return "El LLM no devolvio respuestas."

        content = response.choices[0].message.content
        if isinstance(content, str):
            cleaned = content.strip()
            if cleaned:
                return cleaned

        return "El LLM devolvio una respuesta vacia."

    def _should_fallback_to_local(self, llm_answer: str) -> bool:
        lowered = str(llm_answer or "").lower()
        return any(
            token in lowered
            for token in (
                "insufficient_quota",
                "error code: 429",
                "429",
            )
        )

    def _generate_local_answer(self, query: str, documents: list[Any]) -> str:
        if not documents:
            return (
                "No encontre documentos suficientemente relevantes para responder con evidencia "
                f"a la consulta: {query}."
            )

        selected = documents[: min(3, len(documents))]
        parts: list[str] = []
        prefixes = [
            "Segun la informacion recuperada, ",
            "Ademas, ",
            "Tambien, ",
        ]

        for index, doc in enumerate(selected):
            citation_id = self._get_doc_field(doc, "citation_id", index + 1)
            title = self._to_text(
                self._get_doc_field(doc, "title", f"Documento {index + 1}"),
                f"Documento {index + 1}",
            )
            summary = self._to_text(self._get_doc_field(doc, "summary", ""), "")
            content_text = self._to_text(self._get_doc_field(doc, "content_text", ""), "")
            excerpt = self._best_excerpt(summary, content_text, title).rstrip(".!?")
            prefix = prefixes[index] if index < len(prefixes) else ""
            parts.append(f"{prefix}{excerpt} [{citation_id}].")

        sources = ", ".join(
            f"[{self._get_doc_field(doc, 'citation_id', idx + 1)}] "
            f"{self._to_text(self._get_doc_field(doc, 'title', f'Documento {idx + 1}'), f'Documento {idx + 1}')}"
            for idx, doc in enumerate(selected)
        )
        parts.append(
            "Se devolvio una respuesta local por limite de cuota del proveedor LLM. "
            f"Fuentes usadas: {sources}."
        )
        return " ".join(parts)

    def _best_excerpt(self, summary: str, content_text: str, title: str) -> str:
        text = summary.strip() or content_text.strip() or title.strip()
        text = " ".join(text.split())
        if len(text) <= 280:
            return text
        return text[:277].rstrip() + "..."

    def _get_doc_field(self, doc: Any, field: str, default: Any) -> Any:
        if isinstance(doc, dict):
            return doc.get(field, default)
        return getattr(doc, field, default)

    def _to_text(self, value: Any, default: str) -> str:
        text = str(value or "").strip()
        return text or default

    def _to_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _read_env(self, key: str, default: str) -> str:
        value = os.getenv(key)
        if value is None:
            return default
        cleaned = value.strip()
        return cleaned or default

    def _read_bool_env(self, key: str, *, default: bool) -> bool:
        value = os.getenv(key)
        if value is None:
            return default
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    def _read_int_env(self, key: str, default: int) -> int:
        value = os.getenv(key)
        if value is None:
            return default
        try:
            parsed = int(value.strip())
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _read_float_env(self, key: str, default: float) -> float:
        value = os.getenv(key)
        if value is None:
            return default
        try:
            parsed = float(value.strip())
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= 0 else default

    def _default_model_for_provider(self) -> str:
        if self.provider == "ollama":
            return "llama3.1:8b"
        return "gpt-4o-mini"

    def _default_base_url_for_provider(self) -> str:
        if self.provider == "ollama":
            return "http://localhost:11434"
        return "https://api.openai.com/v1"

    def _is_local_base_url(self) -> bool:
        lowered = self.base_url.lower()
        return "localhost" in lowered or "127.0.0.1" in lowered

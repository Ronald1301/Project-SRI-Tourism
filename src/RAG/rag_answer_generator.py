from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any


class RAGAnswerGenerator:
    """Generador de respuesta RAG usando `ollama run`.

    Recibe query + documentos, construye prompt y genera con `ollama run qwen3`.
    """

    def __init__(self) -> None:
        self.model = self._read_env("RAG_LLM_MODEL", "qwen3")
        self.ollama_cmd = self._read_env("RAG_OLLAMA_CMD", "ollama")
        self.timeout_seconds = self._read_int_env("RAG_LLM_TIMEOUT_SECONDS", 120)
        self.enabled = self._read_bool_env("RAG_LLM_ENABLED", default=True)
        self._init_error = ""

        if shutil.which(self.ollama_cmd) is None:
            self._init_error = (
                f"No se encontro el comando '{self.ollama_cmd}'. "
                "Instala Ollama y verifica que este en PATH."
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
        answer = self._generate_with_ollama_run(active_prompt)

        # Si Ollama falla, evita cortar el flujo y devuelve algo util local.
        if answer.startswith("No se pudo generar la respuesta con Ollama"):
            return self._generate_local_answer(query, documents, error=answer)
        return answer

    def _generate_with_ollama_run(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                [self.ollama_cmd, "run", self.model],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=max(self.timeout_seconds, 5),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return (
                "No se pudo generar la respuesta con Ollama. "
                f"Timeout de {self.timeout_seconds}s para modelo '{self.model}'."
            )
        except Exception as exc:
            return (
                "No se pudo generar la respuesta con Ollama. "
                f"Error ejecutando `{self.ollama_cmd} run {self.model}`: {exc}"
            )

        if result.returncode != 0:
            detail = self._short_error(result.stderr or result.stdout or "sin detalle")
            return (
                "No se pudo generar la respuesta con Ollama. "
                f"Comando `{self.ollama_cmd} run {self.model}` retorno {result.returncode}: {detail}"
            )

        output = (result.stdout or "").strip()
        if output:
            return output
        return "Ollama devolvio una respuesta vacia."

    def _generate_local_answer(self, query: str, documents: list[Any], *, error: str) -> str:
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
            f"{self._to_text(self._get_doc_field(doc, 'title', f'Documento {idx + 1}'), f'Documento {idx + 1}') }"
            for idx, doc in enumerate(selected)
        )
        parts.append(f"Se devolvio respuesta local porque Ollama fallo: {self._short_error(error)}")
        parts.append(f"Fuentes usadas: {sources}.")
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

    def _short_error(self, text: str) -> str:
        normalized = " ".join(str(text or "").split())
        if not normalized:
            return "sin detalle"
        return normalized[:220].rstrip() + ("..." if len(normalized) > 220 else "")

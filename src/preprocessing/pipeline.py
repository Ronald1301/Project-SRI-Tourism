"""
Dataset preprocessing pipeline (multiformato -> tokens) para el SRI.

Lee fuentes en CSV, TXT, MD, HTML, DOCX, detecta/extrae texto y aplica:
cleaning -> tokenization -> stopword removal -> stemming.

Outputs:
- In-memory `documents` mapping: {doc_id: [tokens]}
- Un JSON por fuente procesada en `data/processed/<fuente>.json`:
  { "source": "...", "document_count": n, "documents": [ { "doc_id": "...", "tokens": [...] } ] }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from docx import Document

DEFAULT_EXTENSIONS: Tuple[str, ...] = (".csv", ".txt", ".md", ".html", ".docx")

try:
    from .cleaner import TextCleaner
    from .stemmer import Stemmer
    from .tokenizer import Tokenizer
except ImportError:  # pragma: no cover
    # Allows running as: `python3 src/preprocessing/pipeline.py`
    root_dir = Path(__file__).resolve().parents[2]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.preprocessing.cleaner import TextCleaner
    from src.preprocessing.stemmer import Stemmer
    from src.preprocessing.tokenizer import Tokenizer


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "dataset"


def _normalize_column_name(name: object) -> str:
    if name is None:
        return ""
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().lstrip("\ufeff")
    name = name.lower()
    name = re.sub(r"\s+", "_", name)
    return name


def _candidate_text_columns() -> Tuple[str, ...]:
    return (
        "review",
        "reviews",
        "content",
        "text",
        "review_text",
        "reviewtext",
        "comment",
        "comments",
        "description",
        "message",
        "body",
    )


def detect_text_column(columns: Sequence[object], preferred: Optional[str] = None) -> Optional[str]:
    """
    Returns the best matching column name from the provided list.
    - If `preferred` is provided and exists (case-insensitive), it's used.
    - Otherwise matches against common IR column names.
    """

    if not columns:
        return None

    normalized_to_original: Dict[str, str] = {}
    for col in columns:
        if col is None:
            continue
        col_str = col if isinstance(col, str) else str(col)
        normalized_to_original[_normalize_column_name(col_str)] = col_str

    if preferred:
        pref_norm = _normalize_column_name(preferred)
        if pref_norm in normalized_to_original:
            return normalized_to_original[pref_norm]

    for candidate in _candidate_text_columns():
        cand_norm = _normalize_column_name(candidate)
        if cand_norm in normalized_to_original:
            return normalized_to_original[cand_norm]

    return None


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_md_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_html_file(path: Path) -> str:
    html_text = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(" ", strip=True)


def _read_docx_file(path: Path) -> str:
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)


def guess_text_column_by_length(df: "pd.DataFrame", sample_size: int = 200) -> Optional[str]:
    """
    Heuristic fallback: pick the column whose values have the highest average length.
    Useful when datasets use unexpected column names.
    """

    if df is None or df.empty:
        return None

    best_col = None
    best_avg_len = -1.0

    for col in df.columns:
        try:
            series = df[col].dropna()
        except Exception:
            continue

        if series is None or series.empty:
            continue

        sample = series.astype(str).head(sample_size)
        try:
            avg_len = float(sample.map(len).mean())
        except Exception:
            continue

        if avg_len > best_avg_len:
            best_avg_len = avg_len
            best_col = col

    return best_col


@dataclass
class PreprocessingPipeline:
    language: str = "english"
    cleaner: TextCleaner = field(default_factory=TextCleaner)
    tokenizer: Optional[Tokenizer] = field(default=None)
    stemmer: Optional[Stemmer] = field(default=None)

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(language=self.language)
        if self.stemmer is None:
            self.stemmer = Stemmer(language=self.language)

    def process_text(self, text: object) -> List[str]:
        cleaned = self.cleaner.clean(text)
        tokens = self.tokenizer.tokenize(cleaned)  # type: ignore[union-attr]
        tokens = self.tokenizer.remove_stopwords(tokens)  # type: ignore[union-attr]
        tokens = self.stemmer.stem_tokens(tokens)  # type: ignore[union-attr]
        return [t for t in tokens if t]

    def process_dataframe(
        self,
        df: "pd.DataFrame",
        *,
        text_column: Optional[str] = None,
        doc_id: str = "doc_1",
    ) -> Dict[str, List[str]]:
        if df is None or df.empty:
            return {}

        chosen_col = detect_text_column(list(df.columns), preferred=text_column)
        if not chosen_col:
            chosen_col = guess_text_column_by_length(df)
            if not chosen_col:
                raise ValueError(
                    "No se pudo detectar la columna de texto. "
                    f"Columnas disponibles: {list(df.columns)}"
                )

        if chosen_col not in df.columns:
            # Shouldn't happen, but keeps error messages simple for the user.
            raise ValueError(
                f"La columna '{chosen_col}' no existe en el dataset. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        # Concatenate all rows of the chosen column into a single text blob.
        series = df[chosen_col].dropna().astype(str)
        full_text = "\n".join(series.tolist())
        return {doc_id: self.process_text(full_text)}

    def process_csv(
        self,
        csv_path: Path,
        *,
        text_column: Optional[str] = None,
        doc_id_prefix: Optional[str] = None,
        read_csv_kwargs: Optional[Dict[str, object]] = None,
    ) -> Dict[str, List[str]]:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(str(csv_path))

        kwargs: Dict[str, object] = {}
        if read_csv_kwargs:
            kwargs.update(read_csv_kwargs)

        df = pd.read_csv(csv_path, **kwargs)
        doc_id = doc_id_prefix or "doc_1"
        return self.process_dataframe(df, text_column=text_column, doc_id=doc_id)

    def process_text_blob(
        self,
        text: str,
        *,
        doc_id: str = "doc_1",
    ) -> Dict[str, List[str]]:
        return {doc_id: self.process_text(text)}


def save_processed_documents(documents: Dict[str, List[str]], output_dir: Path) -> None:
    """
    Deprecated: kept for backwards compatibility (per-document JSON).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for doc_id, tokens in documents.items():
        payload = {"doc_id": doc_id, "tokens": tokens}
        out_path = output_dir / f"{doc_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def save_source_documents(source_id: str, documents: Dict[str, List[str]], output_dir: Path) -> Path:
    """
    Save a single JSON per source:
    {
      "source": "<archivo>",
      "document_count": N,
      "documents": [{ "doc_id": ..., "tokens": [...] }]
    }
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": source_id,
        "document_count": len(documents),
        "documents": [{"doc_id": doc_id, "tokens": tokens} for doc_id, tokens in documents.items()],
    }
    out_path = output_dir / f"{source_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def discover_sources(raw_dir: Path, extensions: Sequence[str]) -> List[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []
    ext_set = {e.lower() for e in extensions}
    return sorted([p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in ext_set])


def process_generic_file(path: Path, pipeline: PreprocessingPipeline, text_column: Optional[str] = None) -> Dict[str, List[str]]:
    suffix = path.suffix.lower()
    dataset_id = _slugify(path.stem)

    if suffix == ".csv":
        try:
            return pipeline.process_csv(
                path,
                text_column=text_column,
                doc_id_prefix=f"{dataset_id}_doc",
                read_csv_kwargs={"on_bad_lines": "skip"},
            )
        except TypeError:
            return pipeline.process_csv(path, text_column=text_column, doc_id_prefix=f"{dataset_id}_doc")

    if suffix in {".txt", ".md"}:
        text = _read_text_file(path)
        return pipeline.process_text_blob(text, doc_id=f"{dataset_id}_doc_1")

    if suffix == ".html":
        text = _read_html_file(path)
        return pipeline.process_text_blob(text, doc_id=f"{dataset_id}_doc_1")

    if suffix in {".docx"}:
        text = _read_docx_file(path)
        return pipeline.process_text_blob(text, doc_id=f"{dataset_id}_doc_1")

    raise ValueError(f"Formato no soportado: {suffix}")


def process_all_sources(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    *,
    language: str = "english",
    text_column: Optional[str] = None,
    extensions: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    """
    Procesa todas las fuentes soportadas bajo `raw_dir` y guarda un JSON por archivo en `processed_dir`.
    """

    pipeline = PreprocessingPipeline(language=language)
    all_documents: Dict[str, List[str]] = {}

    ext_list = list(extensions) if extensions else list(DEFAULT_EXTENSIONS)
    sources = discover_sources(raw_dir, ext_list)
    if not sources:
        print(f"[preprocessing] No se encontraron fuentes en: {raw_dir}", file=sys.stderr)
        return {}

    for path in sources:
        dataset_id = _slugify(path.stem)
        try:
            docs = (
                process_generic_file(path, pipeline, text_column=text_column)
            )
        except Exception as exc:
            print(f"[preprocessing] {path.name}: error -> {exc}", file=sys.stderr)
            continue

        save_source_documents(dataset_id, docs, processed_dir)
        all_documents.update(docs)
        print(
            f"[preprocessing] {path.name}: {len(docs)} documentos -> {processed_dir}/{dataset_id}.json",
            file=sys.stderr,
        )

    return all_documents


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocesa fuentes (CSV, TXT, MD, HTML, DOCX) para indexacion.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directorio con fuentes sin procesar.")
    parser.add_argument("--out-dir", default="data/processed", help="Directorio de salida para JSON procesados (1 por fuente).")
    parser.add_argument(
        "--language",
        default="english",
        help="Idioma para stopwords/stemming: english | spanish (en | es).",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Nombre de columna de texto (opcional). Si no se indica, se autodetecta.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    process_all_sources(
        Path(args.raw_dir),
        Path(args.out_dir),
        language=args.language,
        text_column=args.text_column,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

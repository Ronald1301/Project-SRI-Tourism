from __future__ import annotations

from src.vector_db.preset import OUTPUT_DIR, build_vector_db_from_preset, resolve_documents_path


def main() -> int:
    try:
        source_path = resolve_documents_path()
        db = build_vector_db_from_preset()
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    print("Vector DB creada")
    print(f"- Source: {source_path}")
    print(f"- Documentos indexados: {len(db.doc_ids)}")
    print(f"- Output dir: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

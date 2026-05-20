import flet as ft
from src.frontend.utils.highlight import highlight_text

def ResultCard(doc, index, query):
    title = doc.get("title") or "Sin titulo"
    snippet = highlight_text(doc.get("snippet", ""), query)
    score = doc.get("score", 0.0)
    source = doc.get("source") or "Fuente no disponible"
    url = doc.get("url") or ""
    content_type = doc.get("content_type") or "general"

    metadata_parts = [
        f"Score: {score:.4f}",
        f"Tipo: {content_type}",
        f"Fuente: {source}",
    ]
    if url:
        metadata_parts.append(url)

    return ft.Container(
        padding=15,
        margin=5,
        bgcolor="#1a1a1a",
        border_radius=10,
        animate_opacity=300,
        content=ft.Column(
            controls=[
                ft.Text(f"{index + 1}. {title}", weight="bold"),
                ft.Text(snippet or "Sin snippet disponible"),
                ft.Text(" | ".join(metadata_parts), size=11, color="#cbd5e1"),
            ]
        )
    )

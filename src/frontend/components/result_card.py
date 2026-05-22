import flet as ft
from urllib.parse import urlparse

from src.frontend.utils.highlight import highlight_text


def _format_score_percentage(score: float) -> str:
    normalized = max(0.0, min(float(score), 1.0))
    return f"{normalized * 100:.1f}%"


def _pill(label: str, *, bgcolor: str, color: str = "#e5e7eb") -> ft.Container:
    return ft.Container(
        bgcolor=bgcolor,
        border_radius=999,
        padding=ft.Padding(left=10, top=6, right=10, bottom=6),
        content=ft.Text(label, size=11, color=color),
    )


def ResultCard(doc, index, query):
    title = doc.get("title") or "Sin titulo"
    snippet = highlight_text(doc.get("snippet", ""), query)
    score = doc.get("score", 0.0)
    score_percentage = _format_score_percentage(score)
    source = doc.get("source") or "Fuente no disponible"
    url = doc.get("url") or ""
    content_type = doc.get("content_type") or "general"
    location = doc.get("location")
    rating = doc.get("rating")
    domain = urlparse(url).netloc.replace("www.", "") if url else ""

    metadata_controls = [
        _pill(f"Tipo: {content_type}", bgcolor="#273449"),
        _pill(f"Fuente: {source}", bgcolor="#2c2c2c"),
    ]
    if location:
        metadata_controls.append(_pill(f"Ubicacion: {location}", bgcolor="#2f3a2d"))
    if rating:
        metadata_controls.append(_pill(f"Rating: {rating}", bgcolor="#4a3521"))

    return ft.Container(
        padding=18,
        margin=ft.Margin(left=0, top=4, right=0, bottom=4),
        bgcolor="#171717",
        border_radius=14,
        border=ft.Border(
            left=ft.BorderSide(1, "#2a2a2a"),
            top=ft.BorderSide(1, "#2a2a2a"),
            right=ft.BorderSide(1, "#2a2a2a"),
            bottom=ft.BorderSide(1, "#2a2a2a"),
        ),
        animate_opacity=300,
        content=ft.Column(
            spacing=12,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Text(f"#{index + 1}", size=13, color="#94a3b8", weight="bold"),
                        _pill(f"Relevancia: {score_percentage}", bgcolor="#183153", color="#bfdbfe"),
                    ],
                ),
                ft.Text(title, weight="bold", size=18, color="#f8fafc"),
                ft.Text(
                    snippet or "Sin snippet disponible",
                    size=13,
                    color="#d1d5db",
                    max_lines=4,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                ft.ResponsiveRow(
                    controls=[
                        ft.Container(col={"xs": 12, "md": 12}, content=ft.Row(wrap=True, spacing=8, controls=metadata_controls))
                    ]
                ),
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Text(domain or "Sin URL disponible", size=12, color="#93c5fd"),
                        ft.TextButton(
                            "Abrir fuente",
                            url=url if url else None,
                            icon=ft.Icons.OPEN_IN_NEW,
                            disabled=not bool(url),
                        ),
                    ],
                ),
            ]
        )
    )

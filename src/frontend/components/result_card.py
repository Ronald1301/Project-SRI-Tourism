import flet as ft
from urllib.parse import urlparse

from src.frontend.components.feedback_buttons import FeedbackButtons
from src.frontend.utils.highlight import highlight_text


def _format_score_percentage(score: float) -> str:
    normalized = max(0.0, min(float(score), 1.0))
    return f"{normalized * 100:.1f}%"


def _score_badge_colors(score: float) -> tuple[str, str]:
    normalized = max(0.0, min(float(score), 1.0))
    if normalized >= 0.75:
        return "#163a2f", "#bbf7d0"
    if normalized >= 0.45:
        return "#4a3410", "#fde68a"
    return "#4c1d1d", "#fecaca"


def _pill(label: str, *, bgcolor: str, color: str = "#e5e7eb") -> ft.Container:
    return ft.Container(
        bgcolor=bgcolor,
        border_radius=999,
        padding=ft.Padding(left=10, top=6, right=10, bottom=6),
        content=ft.Text(label, size=11, color=color),
    )


def _build_explanation_tile(explanation: dict | None) -> ft.Control | None:
    if not isinstance(explanation, dict):
        return None

    components = explanation.get("components") or []
    boosts = explanation.get("boosts") or []
    penalties = explanation.get("penalties") or []
    exact_matches = explanation.get("exact_matches") or {}

    explanation_rows: list[ft.Control] = [
        ft.Text(
            f"Score final: {float(explanation.get('final_score', 0.0)):.4f} | "
            f"Base: {float(explanation.get('base_score', 0.0)):.4f} | "
            f"Senales: {float(explanation.get('signal_total', 0.0)):.4f}",
            size=12,
            color="#cbd5e1",
        )
    ]

    for component in components[:3]:
        explanation_rows.append(
            ft.Text(
                f"- {component.get('name', 'signal')}: valor={float(component.get('value', 0.0)):.4f}, "
                f"aporte={float(component.get('contribution', 0.0)):.4f}",
                size=12,
                color="#cbd5e1",
            )
        )

    if boosts:
        explanation_rows.append(
            ft.Text(
                "Boosts: " + ", ".join(str(boost.get("name", "")) for boost in boosts[:3]),
                size=12,
                color="#bbf7d0",
            )
        )

    if penalties:
        explanation_rows.append(
            ft.Text(
                "Penalizaciones: " + ", ".join(str(penalty.get("name", "")) for penalty in penalties[:3]),
                size=12,
                color="#fecaca",
            )
        )

    if exact_matches:
        explanation_rows.append(
            ft.Text(
                f"Coincidencia exacta de la consulta: {'si' if exact_matches.get('full_query_phrase') else 'no'}",
                size=12,
                color="#bfdbfe",
            )
        )

    return ft.ExpansionTile(
        title=ft.Text("Ver explicacion de ranking", size=13, color="#cbd5e1"),
        tile_padding=ft.Padding(left=0, top=0, right=0, bottom=0),
        controls_padding=ft.Padding(left=0, top=8, right=0, bottom=0),
        collapsed_bgcolor="#141414",
        bgcolor="#101010",
        controls=explanation_rows,
    )


def ResultCard(
    doc,
    index,
    query,
    page: ft.Page | None = None,
    on_feedback=None,
    on_implicit_feedback=None,
    feedback_value: str | None = None,
    on_feedback_submit=None,
    on_feedback_persist=None,
    on_feedback_success=None,
    on_feedback_error=None,
    initially_hidden: bool = False,
):
    title = doc.get("title") or "Sin titulo"
    snippet = highlight_text(doc.get("snippet", ""), query)
    score = doc.get("score", 0.0)
    score_percentage = _format_score_percentage(score)
    badge_bg, badge_fg = _score_badge_colors(score)
    source = doc.get("source") or "Fuente no disponible"
    url = doc.get("url") or ""
    content_type = doc.get("content_type") or "general"
    location = doc.get("location")
    rating = doc.get("rating")
    explanation = doc.get("explanation")
    domain = urlparse(url).netloc.replace("www.", "") if url else ""

    def copy_url(_):
        if not url or page is None:
            return
        page.run_task(page.clipboard.set, url)
        if callable(on_implicit_feedback):
            on_implicit_feedback(doc.get("doc_id"), "copy_url")
        if callable(on_feedback):
            on_feedback("URL copiada al portapapeles.")

    def open_source(_):
        if callable(on_implicit_feedback):
            on_implicit_feedback(doc.get("doc_id"), "open_source")

    metadata_controls = [
        _pill(f"Tipo: {content_type}", bgcolor="#273449"),
        _pill(f"Fuente: {source}", bgcolor="#2c2c2c"),
    ]
    if location:
        metadata_controls.append(_pill(f"Ubicacion: {location}", bgcolor="#2f3a2d"))
    if rating:
        metadata_controls.append(_pill(f"Rating: {rating}", bgcolor="#4a3521"))
    explanation_tile = _build_explanation_tile(explanation)
    feedback_buttons = FeedbackButtons(
        query=query,
        doc_id=str(doc.get("doc_id") or ""),
        initial_value=feedback_value,
        on_submit=on_feedback_submit,
        on_persist=on_feedback_persist,
        on_success=on_feedback_success,
        on_error=on_feedback_error,
        page=page,
    )

    card = ft.Container(
        padding=20,
        margin=ft.Margin(left=0, top=6, right=0, bottom=6),
        bgcolor="#171717",
        gradient=ft.LinearGradient(colors=["#171717", "#111111"]),
        border_radius=18,
        border=ft.Border(
            left=ft.BorderSide(1, "#2a2a2a"),
            top=ft.BorderSide(1, "#2a2a2a"),
            right=ft.BorderSide(1, "#2a2a2a"),
            bottom=ft.BorderSide(1, "#2a2a2a"),
        ),
        shadow=[
            ft.BoxShadow(
                spread_radius=0,
                blur_radius=18,
                color="#02061733",
                offset=ft.Offset(0, 8),
            )
        ],
        opacity=0 if initially_hidden else 1,
        scale=0.98 if initially_hidden else 1,
        offset=ft.Offset(0, 0.04) if initially_hidden else ft.Offset(0, 0),
        animate_opacity=300,
        animate_scale=220,
        animate_offset=220,
        content=ft.Column(
            spacing=14,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Text(f"#{index + 1}", size=13, color="#94a3b8", weight="bold"),
                        _pill(f"Relevancia: {score_percentage}", bgcolor=badge_bg, color=badge_fg),
                    ],
                ),
                ft.Container(
                    url=url if url else None,
                    ink=bool(url),
                    content=ft.Text(
                        title,
                        weight="bold",
                        size=19,
                        color="#93c5fd" if url else "#f8fafc",
                    ),
                ),
                ft.Text(
                    snippet or "Sin snippet disponible",
                    size=13,
                    color="#d1d5db",
                    max_lines=5,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    selectable=True,
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
                        ft.Text(domain or "Sin URL disponible", size=12, color="#93c5fd", max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                        ft.Row(
                            spacing=0,
                            controls=[
                                feedback_buttons.control,
                                ft.IconButton(
                                    icon=ft.Icons.CONTENT_COPY,
                                    tooltip="Copiar URL",
                                    on_click=copy_url,
                                    disabled=not bool(url),
                                ),
                                ft.TextButton(
                                    "Abrir fuente",
                                    url=url if url else None,
                                    icon=ft.Icons.OPEN_IN_NEW,
                                    on_click=open_source,
                                    disabled=not bool(url),
                                ),
                            ],
                        ),
                    ],
                ),
                explanation_tile if explanation_tile is not None else ft.Container(),
            ]
        )
    )

    def handle_hover(event):
        is_hovering = str(event.data).lower() == "true"
        card.scale = 1.01 if is_hovering else 1
        card.bgcolor = "#1b1b1b" if is_hovering else "#171717"
        card.update()

    card.on_hover = handle_hover
    return card

import flet as ft

from src.frontend.components.loading_indicator import LoadingIndicator
from src.frontend.components.status_text import StatusText
from src.frontend.state import UIState


def _message_container(
    icon: str,
    title: str,
    subtitle: str,
    *,
    bgcolor: str,
    icon_color: str,
    help_text: str | None = None,
) -> ft.Container:
    details = [StatusText(title, subtitle)]
    if help_text:
        details.append(
            ft.Container(
                margin=ft.Margin(left=0, top=6, right=0, bottom=0),
                padding=ft.Padding(left=10, top=8, right=10, bottom=8),
                border_radius=10,
                bgcolor="#121212",
                content=ft.Text(help_text, size=12, color="#dbeafe"),
            )
        )

    return ft.Container(
        bgcolor=bgcolor,
        border_radius=14,
        padding=16,
        border=ft.Border(
            left=ft.BorderSide(1, "#2a2a2a"),
            top=ft.BorderSide(1, "#2a2a2a"),
            right=ft.BorderSide(1, "#2a2a2a"),
            bottom=ft.BorderSide(1, "#2a2a2a"),
        ),
        content=ft.Row(
            spacing=14,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Icon(icon, color=icon_color, size=22),
                ft.Column(spacing=4, controls=details),
            ],
        ),
    )


def _friendly_error_copy(error_message: str | None) -> tuple[str, str, str]:
    raw = str(error_message or "").strip()
    lowered = raw.lower()

    if "connection refused" in lowered or "failed to establish a new connection" in lowered:
        return (
            "No pudimos conectar con el buscador",
            "La interfaz no pudo comunicarse con el servidor local de la aplicacion.",
            "Verifica que la API este encendida con uvicorn y vuelve a intentar la busqueda.",
        )
    if "read timed out" in lowered or "timeout" in lowered:
        return (
            "La busqueda tardo demasiado",
            "El servidor esta tardando mas de lo esperado en responder.",
            "Intenta de nuevo en unos segundos o prueba una consulta mas corta.",
        )
    if "500" in lowered or "internal server error" in lowered:
        return (
            "El servidor encontro un problema",
            "La busqueda llego al backend, pero este no pudo completar la respuesta.",
            "Revisa que el indice y los archivos de documentos del proyecto esten creados antes de buscar.",
        )
    if "no se encontro" in lowered and "documents.jsonl" in lowered:
        return (
            "Faltan datos para poder buscar",
            "La aplicacion aun no tiene cargado el archivo principal de documentos del proyecto.",
            "Ejecuta el crawler o el pipeline para generar los documentos y vuelve a probar.",
        )
    if "consulta vac" in lowered:
        return (
            "Escribe algo para buscar",
            "La caja de busqueda esta vacia y por eso no se pudo iniciar la consulta.",
            "Prueba con un tema, un destino o una pregunta corta sobre turismo.",
        )
    return (
        "No se pudo completar la busqueda",
        "Ocurrio un problema mientras intentabamos recuperar los resultados.",
        "Si el problema continua, revisa que la API este activa y que los datos del proyecto ya esten preparados.",
    )


def StatusBanner(state):

    if state.ui_state == UIState.LOADING:
        return LoadingIndicator(
            getattr(state, "loading_label", "Analizando consulta..."),
            getattr(state, "loading_detail", "Consultando el recuperador y preparando la respuesta."),
            progress=getattr(state, "loading_progress", None),
            history=getattr(state, "loading_history", None),
        )

    if state.ui_state == UIState.OUT_OF_DOMAIN:
        message = "La consulta no está relacionada con el dominio del sistema."
        return _message_container(
            ft.Icons.SEARCH_OFF,
            "Fuera de dominio",
            message,
            bgcolor="#1f1b12",
            icon_color="#fbbf24",
            help_text="Prueba con hoteles, playas, destinos o lugares concretos de Cuba para activar el recuperador.",
        )

    if state.ui_state == UIState.SHOWING_RESULTS and not getattr(state, "results", []):
        return _message_container(
            ft.Icons.SEARCH_OFF,
            "Sin resultados",
            "No encontramos resultados claros para esa consulta.",
            bgcolor="#191919",
            icon_color="#fbbf24",
            help_text="Prueba con otros terminos o un destino mas concreto para obtener evidencias más cercanas.",
        )

    if state.ui_state == UIState.ERROR:
        title, subtitle, help_text = _friendly_error_copy(state.error_message)
        return _message_container(
            ft.Icons.ERROR_OUTLINE,
            title,
            subtitle,
            bgcolor="#261818",
            icon_color="#fca5a5",
            help_text=help_text,
        )

    return None

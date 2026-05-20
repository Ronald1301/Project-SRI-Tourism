import flet as ft

from src.frontend.state import UIState

def StatusBanner(state):

    if state.ui_state == UIState.LOADING:
        return ft.ProgressRing()

    if state.ui_state == UIState.EMPTY:
        return ft.Text("No se encontraron resultados")

    if state.ui_state == UIState.ERROR:
        return ft.Text(f"Error: {state.error_message}", color="red")

    return None

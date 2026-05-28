import flet as ft
import inspect

def SearchBar(on_search):
    last_submitted_query = {"value": ""}
    loading_state = {"value": False}

    query_input = ft.TextField(
        hint_text="Escribe tu consulta...",
        expand=True,
    )

    mode_selector = ft.Dropdown(
        width=170,
        options=[
            ft.dropdown.Option(key="vectorial", text="Vectorial"),
            ft.dropdown.Option(key="lsi", text="LSI"),
            ft.dropdown.Option(key="hybrid_search", text="Hibrido"),
        ],
        value="vectorial"
    )

    topk_selector = ft.Dropdown(
        width=100,
        options=[ft.dropdown.Option(str(i)) for i in [3, 5, 10]],
        value="5"
    )

    search_button = ft.ElevatedButton("Buscar")

    def normalized_query() -> str:
        return (query_input.value or "").strip().casefold()

    def refresh_button_label():
        if loading_state["value"]:
            search_button.text = "Buscando..."
            return
        search_button.text = (
            "Refinar busqueda"
            if normalized_query() and normalized_query() == last_submitted_query["value"]
            else "Buscar"
        )
        try:
            search_button.update()
        except AssertionError:
            pass

    def set_loading(is_loading: bool):
        loading_state["value"] = bool(is_loading)
        query_input.disabled = bool(is_loading)
        mode_selector.disabled = bool(is_loading)
        topk_selector.disabled = bool(is_loading)
        search_button.disabled = bool(is_loading)
        refresh_button_label()
        try:
            query_input.update()
            mode_selector.update()
            topk_selector.update()
            search_button.update()
        except AssertionError:
            pass

    async def trigger_search():
        current_query = normalized_query()
        set_loading(True)
        try:
            result = on_search(
                query_input.value,
                mode_selector.value,
                int(topk_selector.value)
            )
            if inspect.isawaitable(result):
                await result
            if current_query:
                last_submitted_query["value"] = current_query
        finally:
            set_loading(False)
            refresh_button_label()

    async def handle_search(e):
        await trigger_search()

    async def handle_submit(e):
        await trigger_search()

    def handle_query_change(e):
        refresh_button_label()

    query_input.on_submit = handle_submit
    query_input.on_change = handle_query_change
    search_button.on_click = handle_search

    row = ft.Row(
        spacing=12,
        vertical_alignment=ft.CrossAxisAlignment.END,
        controls=[
            query_input,
            mode_selector,
            topk_selector,
            search_button
        ]
    )
    row.set_loading = set_loading
    return row

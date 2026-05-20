import flet as ft

def SearchBar(on_search):

    query_input = ft.TextField(
        hint_text="Escribe tu consulta...",
        expand=True,
    )

    mode_selector = ft.Dropdown(
        width=150,
        options=[
            ft.dropdown.Option("Vectorial"),
            ft.dropdown.Option("LSI"),
            ft.dropdown.Option("RAG"),
        ],
        value="Vectorial"
    )

    topk_selector = ft.Dropdown(
        width=100,
        options=[ft.dropdown.Option(str(i)) for i in [3,5,10]],
        value="5"
    )

    def handle_search(e):
        on_search(
            query_input.value,
            mode_selector.value,
            int(topk_selector.value)
        )

    return ft.Row(
        controls=[
            query_input,
            mode_selector,
            topk_selector,
            ft.ElevatedButton("Buscar", on_click=handle_search)
        ]
    )
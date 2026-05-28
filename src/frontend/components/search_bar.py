import flet as ft

def SearchBar(on_search):

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

    def trigger_search():
        on_search(
            query_input.value,
            mode_selector.value,
            int(topk_selector.value)
        )

    def handle_search(e):
        trigger_search()

    def handle_submit(e):
        trigger_search()

    query_input.on_submit = handle_submit

    return ft.Row(
        spacing=12,
        vertical_alignment=ft.CrossAxisAlignment.END,
        controls=[
            query_input,
            mode_selector,
            topk_selector,
            ft.ElevatedButton("Buscar", on_click=handle_search)
        ]
    )

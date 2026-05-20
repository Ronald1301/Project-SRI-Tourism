import flet as ft

def ResultCard(doc, index):

    return ft.Container(
        padding=15,
        margin=5,
        bgcolor="#1a1a1a",
        border_radius=10,
        content=ft.Column(
            controls=[
                ft.Text(f"{index+1}. {doc.get('title','Sin título')}", weight="bold"),
                ft.Text(doc.get("snippet","")),
                ft.Text(f"Score: {doc.get('score',0):.4f}", size=12),
            ]
        )
    )
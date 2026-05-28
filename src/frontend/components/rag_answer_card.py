import flet as ft


def _meta_pill(label: str, *, bgcolor: str, color: str = "#e5e7eb") -> ft.Container:
    return ft.Container(
        bgcolor=bgcolor,
        border_radius=999,
        padding=ft.Padding(left=10, top=6, right=10, bottom=6),
        content=ft.Text(label, size=11, color=color),
    )


def RagAnswerCard(answer, *, query: str = "", mode: str = "", result_count: int = 0, prompt: str | None = None):

    if not answer:
        return None

    chips = []
    if mode:
        chips.append(_meta_pill(f"Modo: {mode}", bgcolor="#1f2f4a", color="#bfdbfe"))
    if result_count:
        chips.append(_meta_pill(f"Fuentes: {result_count}", bgcolor="#243227", color="#bbf7d0"))

    controls = [
        ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Column(
                    spacing=4,
                    controls=[
                        ft.Text("Respuesta resumida", size=19, weight="bold", color="#f8fafc"),
                        ft.Text(
                            "Sintesis generada a partir del contexto recuperado por el sistema.",
                            size=12,
                            color="#cbd5e1",
                        ),
                    ],
                ),
                ft.Icon(ft.Icons.AUTO_AWESOME, color="#93c5fd", size=24),
            ],
        ),
    ]

    if chips:
        controls.append(ft.Row(wrap=True, spacing=8, controls=chips))

    if query:
        controls.append(
            ft.Text(
                f'Consulta: "{query}"',
                size=12,
                color="#bfdbfe",
                italic=True,
            )
        )

    controls.extend(
        [
            ft.Divider(color="#334155", height=18),
            ft.Text(answer, size=15, color="#e5eefc", selectable=True),
        ]
    )

    if prompt:
        controls.append(
            ft.ExpansionTile(
                title=ft.Text("Ver prompt utilizado", size=13, color="#cbd5e1"),
                tile_padding=ft.Padding(left=0, top=0, right=0, bottom=0),
                controls_padding=ft.Padding(left=0, top=8, right=0, bottom=0),
                collapsed_bgcolor="#17263d",
                bgcolor="#101b2f",
                controls=[
                    ft.Text(prompt, size=12, color="#cbd5e1", selectable=True),
                ],
            )
        )

    return ft.Container(
        padding=22,
        bgcolor="#102235",
        gradient=ft.LinearGradient(colors=["#102235", "#132b45"]),
        border_radius=18,
        shadow=[
            ft.BoxShadow(
                spread_radius=0,
                blur_radius=24,
                color="#02061755",
                offset=ft.Offset(0, 10),
            )
        ],
        border=ft.Border(
            left=ft.BorderSide(1, "#27496d"),
            top=ft.BorderSide(1, "#27496d"),
            right=ft.BorderSide(1, "#27496d"),
            bottom=ft.BorderSide(1, "#27496d"),
        ),
        content=ft.Column(
            spacing=12,
            controls=controls,
        )
    )

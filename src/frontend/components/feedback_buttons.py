from __future__ import annotations

import asyncio
import inspect
from typing import Awaitable, Callable

import flet as ft

FeedbackHandler = Callable[[str], Awaitable[dict] | dict | None]
PersistHandler = Callable[[str | None], None]


class FeedbackButtons:
    def __init__(
        self,
        *,
        query: str,
        doc_id: str,
        initial_value: str | None = None,
        on_submit: FeedbackHandler | None = None,
        on_persist: PersistHandler | None = None,
        on_success: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        page: ft.Page | None = None,
    ) -> None:
        self.query = query
        self.doc_id = doc_id
        self.page = page
        self.on_submit = on_submit
        self.on_persist = on_persist
        self.on_success = on_success
        self.on_error = on_error
        self._value = initial_value if initial_value in {"like", "dislike"} else None
        self._busy = False

        async def _handle_like(_):
            await self._handle_click("like")

        async def _handle_dislike(_):
            await self._handle_click("dislike")

        self.like_button = ft.IconButton(
            tooltip="Marcar como relevante",
            icon=ft.Icons.THUMB_UP_OUTLINED,
            icon_color="#86efac",
            on_click=_handle_like,
            animate_scale=140,
        )
        self.dislike_button = ft.IconButton(
            tooltip="Marcar como no relevante",
            icon=ft.Icons.THUMB_DOWN_OUTLINED,
            icon_color="#fca5a5",
            on_click=_handle_dislike,
            animate_scale=140,
        )
        self.control = ft.Row(
            spacing=0,
            controls=[self.like_button, self.dislike_button],
        )
        self._sync()

    def set_value(self, value: str | None) -> None:
        self._value = value if value in {"like", "dislike"} else None
        self._sync()

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.like_button.disabled = self._busy
        self.dislike_button.disabled = self._busy
        self.control.opacity = 0.72 if self._busy else 1.0
        self._sync()
        self._safe_update(self.control)

    def _sync(self) -> None:
        self.like_button.icon = ft.Icons.THUMB_UP if self._value == "like" else ft.Icons.THUMB_UP_OUTLINED
        self.dislike_button.icon = (
            ft.Icons.THUMB_DOWN if self._value == "dislike" else ft.Icons.THUMB_DOWN_OUTLINED
        )

    async def _pulse(self, button: ft.IconButton) -> None:
        button.scale = 1.08
        self._safe_update(button)
        await asyncio.sleep(0.12)
        button.scale = 1.0
        self._safe_update(button)

    async def _handle_click(self, choice: str) -> None:
        if self._busy:
            return

        previous = self._value
        if previous == choice:
            if self.page is not None:
                self.page.run_task(self._pulse, self.like_button if choice == "like" else self.dislike_button)
            return

        self._value = choice
        self._sync()
        self._safe_update(self.control)

        if self.page is not None:
            self.page.run_task(self._pulse, self.like_button if choice == "like" else self.dislike_button)

        if self.on_submit is None:
            if self.on_persist is not None:
                self.on_persist(choice)
            return

        self.set_busy(True)
        try:
            result = self.on_submit(choice)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(str(result["error"]))
            if self.on_persist is not None:
                self.on_persist(choice)
            if self.on_success is not None:
                self.on_success(choice)
        except Exception as exc:
            self._value = previous
            self._sync()
            self._safe_update(self.control)
            if self.on_persist is not None:
                self.on_persist(previous)
            if self.on_error is not None:
                self.on_error(str(exc))
        finally:
            self.set_busy(False)

    def _safe_update(self, control: ft.Control) -> None:
        try:
            control.update()
        except AssertionError:
            pass

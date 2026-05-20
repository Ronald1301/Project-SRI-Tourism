
from enum import Enum

class UIState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    SUCCESS = "success"
    EMPTY = "empty"
    ERROR = "error"

class AppState:
    def __init__(self):
        self.query = ""
        self.mode = "vectorial"
        self.top_k = 5

        self.results = []
        self.answer_rag = None

        self.ui_state = UIState.IDLE
        self.error_message = None

    def set_loading(self):
        self.ui_state = UIState.LOADING
        self.error_message = None

    def set_success(self, results, answer_rag=None):
        self.results = results
        self.answer_rag = answer_rag
        self.ui_state = UIState.SUCCESS if results else UIState.EMPTY

    def set_error(self, message):
        self.ui_state = UIState.ERROR
        self.error_message = message
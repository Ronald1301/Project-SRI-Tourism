
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
        self.prompt = None
        self.expansion_info = None

        self.page = 1
        self.has_more = False

        self.ui_state = UIState.IDLE
        self.error_message = None
        self.loading_label = "Cargando."

    def set_loading(self):
        self.ui_state = UIState.LOADING
        self.error_message = None
        self.loading_label = "Cargando."

    def set_success(self, results, answer_rag=None, prompt=None, has_more=False, expansion_info=None):
        self.results = list(results)
        self.answer_rag = answer_rag
        self.prompt = prompt
        self.expansion_info = expansion_info
        self.has_more = has_more
        self.ui_state = UIState.SUCCESS if results else UIState.EMPTY

    def set_error(self, message):
        self.ui_state = UIState.ERROR
        self.error_message = message

    def reset_search(self, query, mode, top_k):
        self.query = query
        self.mode = mode
        self.top_k = top_k
        self.page = 1
        self.results = []
        self.answer_rag = None
        self.prompt = None
        self.expansion_info = None
        self.has_more = False
        self.error_message = None

    def append_results(self, new_results):
        self.results.extend(new_results)

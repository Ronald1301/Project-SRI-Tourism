
from enum import Enum


class UIState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    SHOWING_RESULTS = "showing_results"
    OUT_OF_DOMAIN = "out_of_domain"
    ERROR = "error"
    EMPTY = "showing_results"


class AppState:
    def __init__(self):
        self.query = ""
        self.top_k = 5

        self.results = []
        self.answer_rag = None
        self.prompt = None
        self.expansion_info = None
        self.domain_info = None
        self.feedback_by_result: dict[str, str] = {}

        self.ui_state = UIState.IDLE
        self.error_message = None
        self.loading_stage = "idle"
        self.loading_label = "Cargando."
        self.loading_detail = "Preparando la consulta."

    def set_loading(self):
        self.ui_state = UIState.LOADING
        self.error_message = None
        self.loading_stage = "checking_domain"
        self.loading_label = "Analizando consulta..."
        self.loading_detail = "Verificando el dominio antes de buscar."

    def set_success(self, results, answer_rag=None, prompt=None, expansion_info=None, domain_info=None):
        self.results = list(results)
        self.answer_rag = answer_rag
        self.prompt = prompt
        self.expansion_info = expansion_info
        self.domain_info = domain_info
        self.ui_state = UIState.SHOWING_RESULTS
        self.loading_stage = "done"
        self.loading_label = "Listo."
        self.loading_detail = "La respuesta ya fue recibida."

    def set_out_of_domain(self, message=None, domain_info=None):
        self.results = []
        self.answer_rag = None
        self.prompt = None
        self.expansion_info = None
        self.domain_info = domain_info
        self.ui_state = UIState.OUT_OF_DOMAIN
        self.error_message = message
        self.loading_stage = "out_of_domain"
        self.loading_label = "Fuera de dominio."
        self.loading_detail = "La consulta no corresponde al sistema."

    def set_error(self, message):
        self.ui_state = UIState.ERROR
        self.error_message = message
        self.loading_stage = "done"

    def reset_search(self, query, top_k):
        self.query = query
        self.top_k = top_k
        self.results = []
        self.answer_rag = None
        self.prompt = None
        self.expansion_info = None
        self.domain_info = None
        self.feedback_by_result = {}
        self.error_message = None
        self.loading_stage = "idle"
        self.loading_label = "Cargando."
        self.loading_detail = "Preparando la consulta."

    def append_results(self, new_results):
        self.results.extend(new_results)

    def feedback_key(self, query: str, doc_id: str) -> str:
        return f"{(query or '').strip().casefold()}::{(doc_id or '').strip()}"

    def get_feedback_choice(self, query: str, doc_id: str) -> str | None:
        return self.feedback_by_result.get(self.feedback_key(query, doc_id))

    def set_feedback_choice(self, query: str, doc_id: str, choice: str | None):
        key = self.feedback_key(query, doc_id)
        if choice:
            self.feedback_by_result[key] = choice
        else:
            self.feedback_by_result.pop(key, None)

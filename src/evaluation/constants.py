from pathlib import Path


DEFAULT_QRELS_PATH = Path("data/evaluation/rec01_qrels.json")
DEFAULT_REPORT_PATH = Path("data/evaluation/reports/eval_report.json")
DEFAULT_MARKDOWN_REPORT_PATH = Path("data/evaluation/reports/eval_report.md")

SUPPORTED_SYSTEMS = [
    "lsi_baseline",
    "lsi_refined",
    "lsi_expanded",
    "vectorial",
    "hybrid_search",
]
BASELINE_SYSTEM = "lsi_baseline"

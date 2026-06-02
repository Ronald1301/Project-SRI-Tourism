# SRI Tourism - Sistema de Recuperación de Información

Sistema de recuperación de información para reseñas y consultas de turismo en Cuba.

El proyecto combina una base original de recuperación con el estado actual del sistema, que ya incluye arquitectura híbrida, detección de dominio, expansión de consultas, búsqueda web, RAG, evaluación offline y frontend reactivo en Flet.

## Estructura actual del proyecto

```text
Project-SRI-Tourism/
├── Datos.txt
├── Dockerfile
├── Orientacion.pdf
├── Pipelines/
│   ├── Corte 2.txt
│   ├── Corte 3.txt
│   ├── Pipeline completo.txt
│   ├── Roadmap_frontend.md
│   ├── corte 1.txt
│   └── pipelin final.txt
├── Preguntas-Video.txt
├── README.md
├── Reports/
│   ├── Final Report/
│   │   ├── Informe.tex
│   │   ├── informe2/
│   │   │   ├── informe2.tex
│   │   │   └── ...
│   │   └── ...
│   ├── Report I/
│   │   ├── Report I.tex
│   │   └── ...
│   └── rank/
│       └── DOC-RANK-01.tex
├── data/
│   ├── config/
│   │   ├── domain_detection.json
│   │   ├── evaluation.json
│   │   ├── query_expansion.json
│   │   └── tourism_synonyms.json
│   ├── evaluation/
│   │   ├── rec01_qrels.json
│   │   ├── reports/
│   │   │   ├── eval_report.json
│   │   │   └── eval_report.md
│   │   └── results/
│   │       ├── audit_comparison.json
│   │       ├── audit_comparison.md
│   │       ├── eval_report.json
│   │       └── eval_report.md
│   ├── feedback/
│   │   ├── query_feedback.db
│   │   └── query_feedback.json
│   ├── index/
│   │   ├── doc_vectors.npy
│   │   ├── lsi_metadata.json
│   │   ├── lsi_model.pkl
│   │   ├── tfidf_matrix.npz
│   │   ├── tfidf_meta.json
│   │   └── vocabulary.json
│   ├── processed/
│   │   ├── lsi_training/
│   │   │   ├── baracoa.json
│   │   │   ├── cayo_guillermo.json
│   │   │   ├── documents.json
│   │   │   └── ...
│   │   └── vector_db/
│   │       ├── embeddings.npy
│   │       ├── faiss.index
│   │       ├── index_to_doc_id.json
│   │       └── meta.json
│   └── raw/
│       ├── baracoa.txt
│       ├── cayo_guillermo.md
│       ├── la_habana.md
│       ├── matanzas.md
│       ├── pinar_del_rio.txt
│       ├── trinidad.md
│       ├── turismo_habana.txt
│       ├── turismo_ruta.md
│       └── web_documents.jsonl
├── main.py
├── requirements.txt
├── scripts/
│   └── migrate_feedback_json_to_sqlite.py
├── src/
│   ├── RAG/
│   │   ├── DOCUMENTACION_TECNICA_RAG.md
│   │   ├── rag_answer_generator.py
│   │   └── rag_pipeline.py
│   ├── api/
│   │   ├── bootstrap.py
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── models.py
│   │   └── service/
│   │       ├── __init__.py
│   │       └── rag_service.py
│   ├── evaluation/
│   │   ├── cli.py
│   │   ├── constants.py
│   │   ├── experiment_runner.py
│   │   ├── metrics.py
│   │   ├── qrels.py
│   │   ├── report_generator.py
│   │   ├── systems.py
│   │   └── __init__.py
│   ├── frontend/
│   │   ├── app.py
│   │   ├── api/
│   │   │   ├── client.py
│   │   │   └── mock_client.py
│   │   ├── components/
│   │   │   ├── feedback_buttons.py
│   │   │   ├── loading_indicator.py
│   │   │   ├── rag_answer_card.py
│   │   │   ├── result_card.py
│   │   │   ├── search_bar.py
│   │   │   ├── status_banner.py
│   │   │   └── status_text.py
│   │   ├── icons/
│   │   │   ├── avion.ico
│   │   │   ├── avion1.ico
│   │   │   ├── avion2.ico
│   │   │   ├── cuba1.ico
│   │   │   ├── cuba2.ico
│   │   │   ├── cuba3.ico
│   │   │   ├── cuba4.ico
│   │   │   └── turismo.ico
│   │   ├── models.py
│   │   ├── state.py
│   │   ├── theme.py
│   │   ├── utils/
│   │   │   └── highlight.py
│   │   └── views/
│   │       └── search_page.py
│   ├── indexing/
│   │   ├── inverted_index.py
│   │   ├── tfidf_index.py
│   │   └── __init__.py
│   ├── preprocessing/
│   │   ├── cleaner.py
│   │   ├── pipeline.py
│   │   ├── stemmer.py
│   │   ├── tokenizer.py
│   │   ├── README.md
│   │   └── __init__.py
│   ├── retrieval/
│   │   ├── domain_detector.py
│   │   ├── evaluate.py
│   │   ├── lsi_model.py
│   │   ├── query_expansion.py
│   │   ├── QUERY_EXPANSION_GUIDE.md
│   │   ├── ranking_signals.py
│   │   ├── reranker.py
│   │   ├── search.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── corpus_checker.py
│   │   ├── file_manager.py
│   │   ├── system_state.py
│   │   └── __init__.py
│   └── web_crawler/
│       ├── DOCUMENTACION_TECNICA.md
│       ├── config.py
│       ├── crawler.py
│       ├── data_ingestion.py
│       ├── insufficiency_policy.py
│       ├── policies.py
│       ├── run.py
│       ├── scraper.py
│       ├── sites.py
│       ├── storage.py
│       ├── url_importance_policy.py
│       ├── visited_urls.txt
│       ├── web_search_client.py
│       └── __init__.py
├── tests/
│   ├── test_audit_integration.py
│   ├── test_evaluate_metrics.py
│   ├── test_feedback_bias.py
│   └── test_query_expansion.py
└── ...
```

## Requisitos

- Python 3.12 o superior
- `venv`
- Dependencias del proyecto instaladas
- Ollama local si quieres usar el fallback LLM

## Instalación local

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si ejecutas `python3 main.py ...` desde la raíz del proyecto y existe `.venv/`, el script se relanza automáticamente usando `.venv/bin/python`.

## Ejecución local

### Backend

```powershell
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```powershell
python src/frontend/app.py
```

Si el frontend debe apuntar a otra API:

```powershell
$env:SRI_API_BASE_URL="http://127.0.0.1:8000"
python src/frontend/app.py
```

## Docker

El proyecto mantiene compatibilidad con Docker y Docker Compose.

Construir la imagen:

```bash
docker build -t sri-tourism .
```

Nota: `data/` está excluido en `.dockerignore` para mantener la imagen liviana.

### Docker Compose

La composición separa el sistema en tres contenedores:

- `ollama`: servidor de inferencia para la generación RAG.
- `backend`: API FastAPI + recuperación + RAG.
- `frontend`: interfaz visual Flet que consume la API.

Construir ambos servicios:

```bash
docker compose build
```

Levantar la API y el frontend:

```bash
docker compose up
```

La API quedará disponible en:

- `http://localhost:8000`

Ollama quedará disponible en:

- `http://localhost:11435`

La interfaz visual quedará disponible en:

- `http://localhost:8550`

Si necesitas reconstruir la base vectorial y los artefactos LSI al arrancar la API, asegúrate de montar `data/` como volumen y tener disponible el corpus inicial en el host.
La primera vez que levantes `docker compose up` puede tardar más porque el contenedor de Ollama descarga el modelo `qwen3` en su volumen persistente.

## Endpoints principales

- `GET /health`
- `POST /search`
- `GET /query-stream`
- `POST /feedback`
- `POST /feedback/implicit`

## Módulos implementados

- **web_crawler**: descarga y extrae contenido turístico desde distintas fuentes para alimentar el corpus.
- **vector_db**: construye y persiste la base vectorial con embeddings, metadatos y FAISS.
- **preprocessing**: limpia, normaliza, tokeniza y reduce el texto antes de indexar o buscar.
- **indexing**: genera los artefactos de TF-IDF y LSI usados en recuperación.
- **retrieval**: ejecuta la búsqueda híbrida, expansión de consultas, detección de dominio, feedback y reranking.
- **RAG**: construye el prompt, selecciona evidencia y genera la respuesta final.
- **api**: expone FastAPI, integra búsqueda, streaming SSE, feedback y serialización de respuestas.
- **frontend**: presenta la interfaz Flet, consume la API y refleja estados en tiempo real.
- **evaluation**: calcula Precision, Recall, MAP, MRR, NDCG, R-Precision y compara baseline vs refinado.
- **utils**: agrupa utilidades auxiliares compartidas por el sistema.

## Explicación breve de los módulos

- **Detección de dominio**: filtra consultas fuera de turismo en Cuba antes de ejecutar búsqueda costosa.
- **Expansión de consulta**: amplía la query con PRF, Rocchio, sinónimos y n-gramas sin perder la original.
- **Feedback**: guarda likes/dislikes e interacciones implícitas para ajustar el ranking futuro.
- **Búsqueda local**: recupera evidencia con señales vectoriales, LSI y RRF.
- **Búsqueda web**: se activa solo cuando el corpus local no es suficiente.
- **Generación RAG**: redacta la respuesta final usando la evidencia recuperada.
- **Evaluación offline**: mide la calidad del sistema con métricas clásicas de IR y reportes comparativos.
- **Frontend reactivo**: muestra estados de proceso, resultados, errores y feedback visual.

## Notas técnicas

- El detector de dominio se ejecuta antes de la búsqueda.
- La búsqueda web solo se activa cuando el corpus local no basta.
- El frontend consume estados en tiempo real por SSE.
- El feedback se persiste en SQLite.
- El modelo de Ollama se define en `src/api/config.py`.

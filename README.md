# SRI Tourism - Information Retrieval System

Sistema de recuperación de información para reseñas de turismo.

## Estructura del Proyecto

```
sri_turismo/
├── data/              # Directorio de datos
│   ├── raw/          # Datos sin procesar
│   ├── processed/    # Datos procesados
│   └── index/        # Índices generados
├── src/              # Código fuente
│   ├── web_crawler/  # Web crawling / scraping
│   ├── vector_db/    # Base de datos vectorial inicial
│   ├── preprocessing/# Preprocesamiento de texto
│   ├── indexing/     # Creación de índices
│   ├── retrieval/    # Recuperación de documentos
│   └── utils/        # Utilidades
├── main.py           # Orquestador de pruebas
├── requirements.txt  # Dependencias
└── README.md        # Este archivo
```

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si ejecutas `python3 main.py ...` desde la raíz del proyecto y existe `.venv/`, el script se relanza automáticamente usando `.venv/bin/python`.

## Docker

Construir la imagen:

```bash
docker build -t sri-tourism .
```

Ejecutar con volumen para datos (recomendado):

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  sri-tourism python3 main.py pipeline
```

Nota: `data/` está excluido en `.dockerignore` para mantener la imagen liviana.

Ejecutar consultas:

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  sri-tourism python3 main.py lsi_query "turismo en cuba" --top-k 5
```

Evaluar baseline vs recuperador refinado:

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  sri-tourism python3 main.py evaluate_rec01 --top-k 5
```

Consultar el RAG:

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  sri-tourism python3 main.py rag_query "playas en cuba" --top-k 4
```

### Docker Compose

La composición ahora separa el sistema en tres contenedores:

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

- `http://localhost:11434`

La interfaz visual quedará disponible en:

- `http://localhost:8550`

Si necesitas reconstruir la base vectorial y los artefactos LSI al arrancar la API, asegúrate de montar `data/` como volumen y tener disponible el corpus inicial en el host.
La primera vez que levantes `docker compose up` puede tardar más porque el contenedor de Ollama descarga el modelo `qwen3` en su volumen persistente.

Si quieres ejecutar comandos puntuales del backend desde Compose, puedes usar:

```bash
docker compose run --rm backend python3 main.py lsi_query "turismo en cuba" --top-k 5
```

## Uso

Para probar el flujo integrado:

```bash
python3 main.py pipeline
```

Nota: `pipeline` solo ejecuta `crawl + vectordb`. No construye el indice clasico TF-IDF/LSI ni muestra ranking.

Para habilitar consultas LSI y la evaluacion offline, primero debes entrenar el indice:

```bash
python3 main.py lsi_train
```

Comandos individuales:

```bash
python3 main.py crawl
python3 main.py vectordb
python3 main.py pipeline
python3 main.py query "playas en cuba" --top-k 5
python3 main.py rag_query "playas en cuba" --top-k 4
python3 main.py rag_query "playas en cuba" --top-k 4 --show-prompt
python3 main.py lsi_train
python3 main.py lsi_query "turismo en cuba" --top-k 5
python3 main.py evaluate_rec01 --top-k 5
python3 main.py web_search "playas cubanas" --top-k 10 --output data/raw/web_search/documents.jsonl
```

Comandos que devuelven ranking:

```bash
python3 main.py query "playas en cuba" --top-k 5
python3 main.py lsi_query "turismo en cuba" --top-k 5
python3 main.py evaluate_rec01 --top-k 5
```

- `query`: ranking de la base vectorial (`vector_db`).
- `lsi_query`: ranking del recuperador clasico refinado (`TF-IDF + LSI + rerank + threshold`).
- `evaluate_rec01`: compara el ranking baseline vs refinado con `P@3`, `P@5`, `MAP` y `NDCG@5`.

Comandos RAG:

```bash
python3 main.py rag_query "playas en cuba" --top-k 4
python3 main.py rag_query "playas en cuba" --top-k 4 --show-prompt
```

- `rag_query`: recupera documentos desde la base vectorial y construye una respuesta RAG basada en el contexto recuperado.
- `--show-prompt`: imprime el prompt completo construido para inspeccion o depuracion.

Entradas directas por modulo:

```bash
python3 -m  src.web_crawler.run
```

El crawler usa valores por defecto definidos en `src/web_crawler/sites.py`.
Edita ese archivo para cambiar seeds, dominios, limites y politicas por sitio.

### Ejecución del frontend en local

```bash
python3 src/frontend/app.py
```

Si el frontend debe apuntar a otra API, define:

```bash
export SRI_API_BASE_URL=http://127.0.0.1:8000
```

Para construir la base de datos vectorial inicial:

```bash
python3 src/vector_db/run.py
```

Configura la entrada en `src/vector_db/preset.py`.

## Módulos

- **web_crawler**: Descarga y extrae reseñas de turismo
- **vector_db**: Almacena embeddings (Sentence Transformers) y metadatos iniciales
- **preprocessing**: Limpieza y normalización de texto
- **indexing**: Creación de índices de búsqueda
- **retrieval**: Motor de búsqueda y recuperación
- **utils**: Funciones auxiliares

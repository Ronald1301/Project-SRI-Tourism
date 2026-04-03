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
pip install -r requirements.txt
```

## Uso

Para probar el flujo integrado:

```bash
python3 main.py pipeline
```

Comandos individuales:

```bash
python3 main.py crawl
python3 main.py vectordb
python3 main.py query "playas en cuba" --top-k 5
python3 main.py lsi_train
python3 main.py lsi_query "turismo en cuba" --top-k 5
```

Entradas directas por modulo:

```bash
python3 -m  src.web_crawler.run
```

El crawler usa valores por defecto definidos en `src/web_crawler/sites.py`.
Edita ese archivo para cambiar seeds, dominios, limites y politicas por sitio.

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

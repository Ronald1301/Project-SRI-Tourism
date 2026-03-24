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
├── main.py           # Punto de entrada
├── requirements.txt  # Dependencias
└── README.md        # Este archivo
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python3 main.py crawl
```

El crawler usa valores por defecto definidos en `src/web_crawler/preset.py`.
Edita ese archivo para cambiar seeds, dominios, limites y politicas.

Para construir la base de datos vectorial inicial:

```bash
python3 main.py vectordb
```

Configura la entrada en `src/vector_db/preset.py`.

## Módulos

- **web_crawler**: Descarga y extrae reseñas de turismo
- **vector_db**: Almacena embeddings (Sentence Transformers) y metadatos iniciales
- **preprocessing**: Limpieza y normalización de texto
- **indexing**: Creación de índices de búsqueda
- **retrieval**: Motor de búsqueda y recuperación
- **utils**: Funciones auxiliares

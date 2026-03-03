# SRI Tourism - Information Retrieval System

Sistema de recuperación de información para reseñas de turismo.

## Estructura del Proyecto

```
sri_turismo/
├── data/              # Directorio de datos
│   ├── raw/          # Datos sin procesar
│   ├── processed/    # Datos procesados
│   └── index/        # Índices generados
├── crawler/          # Web scraping
├── preprocessing/    # Preprocesamiento de texto
├── indexing/         # Creación de índices
├── retrieval/        # Recuperación de documentos
├── utils/            # Utilidades
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
python main.py
```

## Módulos

- **crawler**: Descarga reseñas de turismo
- **preprocessing**: Limpieza y normalización de texto
- **indexing**: Creación de índices de búsqueda
- **retrieval**: Motor de búsqueda y recuperación
- **utils**: Funciones auxiliares

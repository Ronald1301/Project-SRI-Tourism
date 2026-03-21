# Preprocessing (SRI Turismo)

Este paquete implementa el **modulo de preprocesamiento** del pipeline clasico de Recuperacion de Informacion:

Adquisicion -> Preprocesamiento -> Indexacion -> Recuperacion

El objetivo es transformar texto crudo (CSV, TXT, MD, HTML, DOCX) en una lista de **tokens normalizados** lista para indexar.

## Por que existe `pipeline.py`?

Los modulos `cleaner.py`, `tokenizer.py` y `stemmer.py` son piezas pequenas y reutilizables.
Pero para cumplir el flujo completo del proyecto (descubrir archivos -> detectar/extraer texto -> procesar documentos -> guardar JSON en `data/processed/`), hace falta una capa que **orqueste** esos pasos y maneje multiples formatos y errores. Esa capa es `pipeline.py` y se expone desde `preprocessing/__init__.py`.

## Modulos

### `cleaner.py`

- `TextCleaner.clean(text) -> str`: normaliza el texto para tokenizar.
- Pasos: minusculas, eliminar HTML, eliminar puntuacion/numeros/caracteres especiales, colapsar espacios.
- Incluye normalizacion tipica de IR: remocion de diacriticos (acentos) para unificar variantes: `habitaci\u00f3n -> habitacion`.

### `tokenizer.py`

- `Tokenizer.tokenize(text) -> List[str]`: divide por espacios y filtra tokens vacios y muy cortos (`< 2`).
- `Tokenizer.remove_stopwords(tokens) -> List[str]`: elimina stopwords.
- Soporta stopwords en:
  - Ingles (`language="english"` / `en`)
  - Espanol (`language="spanish"` / `es`)
- Si el corpus de stopwords de NLTK no esta disponible, usa una lista fallback incluida en el proyecto.

### `stemmer.py`

- `Stemmer(language=...)` aplica stemming segun idioma:
  - Ingles: `PorterStemmer` (NLTK)
  - Espanol: `SnowballStemmer("spanish")` (NLTK)

### `pipeline.py` (orquestador)

Provee el pipeline end-to-end:

- Descubre multiples fuentes en `data/raw/` (CSV, TXT, MD, HTML, DOCX; se detectan automaticamente).
- Detecta automaticamente la columna de texto (por nombres tipicos como `review`, `text`, `review_text`, o por heuristica de longitud).
- Procesa cada documento (para CSV se junta todo el texto de la columna detectada; para TXT/MD/HTML/DOCX es el archivo completo): `clean -> tokenize -> stopwords -> stemming`.
- Devuelve el diccionario esperado por el indexador: `{doc_id: [tokens]}`.
- Guarda **un solo JSON por fuente** en `data/processed/<nombre_fuente>.json`:

```json
{
  "source": "reviews",
  "document_count": 3,
  "documents": [
    {"doc_id": "reviews_doc_1", "tokens": ["hotel", "clean", "room", "good", "servic"]},
    {"doc_id": "reviews_doc_2", "tokens": ["beach", "resort", "beauti", "view"]},
    {"doc_id": "reviews_doc_3", "tokens": ["food", "poor"]}
  ]
}
```

## Ejecucion

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el pipeline sobre todas las fuentes soportadas:

```bash
python3 -m preprocessing.pipeline --raw-dir data/raw --out-dir data/processed --language english
```

Para espanol:

```bash
python3 -m preprocessing.pipeline --language spanish
```

Opciones utiles:

- `--language spanish` para stopwords/stemming en espanol.

Notas:

- Si una columna de texto en CSV no puede detectarse, el pipeline reporta el error y sigue con la siguiente fuente.
- Para CSV se toma la columna de texto, se concatenan todas sus filas y se procesa como un solo documento; para TXT/MD/HTML/DOCX se procesa el archivo completo como un solo documento.
- Cada fuente produce **un** archivo JSON en `data/processed/`, evitando miles de archivos sueltos.

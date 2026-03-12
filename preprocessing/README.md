# Preprocessing (SRI Turismo)

Este paquete implementa el **modulo de preprocesamiento** del pipeline clasico de Recuperacion de Informacion:

Adquisicion -> Preprocesamiento -> Indexacion -> Recuperacion

El objetivo es transformar texto crudo (por ejemplo resenas en CSV) en una lista de **tokens normalizados** lista para indexar.

## Por que existe `pipeline.py`?

Los modulos `cleaner.py`, `tokenizer.py` y `stemmer.py` son piezas pequenas y reutilizables.
Pero para cumplir el flujo completo del proyecto (leer CSV -> detectar columna -> procesar cada documento -> guardar JSON en `data/processed/`), hace falta una capa que **orqueste** esos pasos y maneje multiples datasets y errores.
Esa capa es `pipeline.py`.

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

### `pipeline.py`

Provee el pipeline end-to-end:

- Descubre multiples CSV en `data/raw/` (recursivo).
- Detecta automaticamente la columna de texto (por nombres tipicos como `review`, `text`, `review_text`, o por heuristica de longitud).
- Procesa cada fila: `clean -> tokenize -> stopwords -> stemming`.
- Devuelve el diccionario esperado por el indexador: `{doc_id: [tokens]}`
- Guarda cada documento procesado como JSON en `data/processed/`:

```json
{
  "doc_id": "reviews_doc_1",
  "tokens": ["hotel", "clean", "room", "good", "servic"]
}
```

## Ejecucion

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el pipeline sobre todos los CSV:

```bash
python3 -m preprocessing.pipeline --raw-dir data/raw --out-dir data/processed --language english
```

Para espanol:

```bash
python3 -m preprocessing.pipeline --language spanish
```

Notas:
- Si una columna de texto no puede detectarse, el pipeline reporta el error y sigue con el siguiente dataset.
- Los JSON generados quedan en `data/processed/` (un archivo por documento).
  - Importante: el pipeline asume `1 fila del CSV = 1 documento`. Por eso, si un dataset tiene 10,000 filas, se crean ~10,000 archivos JSON.

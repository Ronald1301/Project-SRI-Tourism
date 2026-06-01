# Query Expansion Guide

Este modulo enriquece consultas turisticas antes de la recuperacion final.

## Configuracion

- `data/config/query_expansion.json`
  - `max_terms`: numero maximo de terminos a añadir
  - `top_documents_for_context`: cuantos documentos alimentan la expansion
  - `acceptance_threshold`: umbral de comparacion para aceptar expansion
  - `score_floor`: score minimo aceptado por candidato
  - `synonym_limit_per_term`: maximo de sinónimos por termino
  - `cache_max_size` y `cache_ttl_seconds`: caché de expansiones
  - `enable_ngrams`: activa 1-2-3 gramas
  - `ngram_unigram_weight`, `ngram_bigram_weight`, `ngram_trigram_weight`: pesos controlados de n-gramas
  - `logging_level`: nivel de logging del expander

- `data/config/tourism_synonyms.json`
  - diccionario turistico con sinónimos bidireccionales
  - admite claves con guiones bajos para frases, por ejemplo `casco_historico`

## Flujo

1. Se normaliza la consulta.
2. Se calcula una clave de caché con query, documentos de contexto y feedback.
3. Se añaden candidatos desde:
   - pseudo relevancia
   - Rocchio
   - sinónimos
   - coocurrencia
   - feedback explícito e implícito
   - n-gramas
4. Se ordenan, se deduplican y se recorta a `max_terms`.
5. Se compara la expansión con el resultado original usando `acceptance_threshold`.

## Feedback

El feedback operativo se guarda solo en SQLite en `data/feedback/query_feedback.db`.
Si necesitas migrar historial viejo desde `data/feedback/query_feedback.json`, ejecuta `scripts/migrate_feedback_json_to_sqlite.py` una vez.

## Logs

El expander emite trazas estructuradas por tecnica:

- `synonyms`
- `pseudo_relevance`
- `cooccurrence`
- `feedback`
- `rocchio`
- `ngrams`

Cada `ExpansionResult` incluye un campo `trace` y un indicador `cached`.

## Notas practicas

- Para que la busqueda web use la consulta expandida, cambia `web_query = query` por `web_query = selected_query` en `src/api/service/rag_service.py`.
- Por defecto, la parte web se alimenta con la consulta original cuando la evidencia local es insuficiente.

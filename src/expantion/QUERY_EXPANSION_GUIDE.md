# Query Expansion Guide

Este modulo implementa `OPT-01`: expansion de consultas y retroalimentacion por relevancia.

## Ubicacion

- Codigo principal: `src/expantion/query_expander.py`
- Configuracion: `src/config/expansion_config.json`
- Sinonimos bilingues: `src/expantion/synonyms_tourism.json`
- Feedback SQLite: `data/feedback/query_feedback.db`
- Wrapper retrocompatible: `src/retrieval/query_expansion.py`

## Flujo

1. El usuario envia una consulta.
2. El backend recupera los primeros documentos candidatos.
3. `QueryExpander.expand_query()` combina varias tecnicas.
4. Se construye una consulta enriquecida.
5. El backend compara resultados crudos vs expandidos.
6. Solo acepta la expansion si no degrada el ranking por debajo del umbral configurado.

## Tecnicas implementadas

- Sinonimos: diccionario turistico espanol/ingles con expansion bidireccional.
- Pseudo relevance feedback: extrae terminos de los primeros documentos recuperados.
- N-gramas: usa unigramas, bigramas y trigramas con multiplicadores configurables.
- Co-ocurrencia: agrega palabras cercanas a terminos de la consulta dentro de los documentos top.
- Feedback explicito: usa documentos marcados como relevantes o no relevantes.
- Feedback implicito: usa interacciones como abrir fuente o copiar URL.
- Rocchio: ajusta el vector de consulta con documentos positivos y negativos.

## Configuracion importante

`global.acceptance_threshold` controla cuanto debe conservar la expansion respecto a la consulta original.
El valor por defecto es `0.75`, mas conservador que el valor anterior `0.65`.

`global.max_terms_to_add` limita el crecimiento de la consulta para evitar drift semantico.
El valor por defecto es `5`.

## Feedback

El feedback explicito se registra desde `/feedback`.
El feedback implicito se registra desde `/feedback/implicit`.

El esquema SQLite mantiene:

- `feedback_explicit`
- `feedback_implicit`

La tabla implicita tiene una restriccion unica por `query_hash`, `doc_id` y `event_group`.
Esto evita contar dos veces acciones equivalentes como abrir fuente y copiar URL.

## Migracion desde JSON

```powershell
python scripts/migrate_feedback_json_to_sqlite.py
```

## Ejemplo

Consulta:

```text
beaches near Havana
```

Expansion esperada:

```text
beaches near Havana playas habana costa seaside cayo
```

La salida real depende de los documentos top recuperados y del feedback existente.

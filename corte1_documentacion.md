# Documentacion Corte 1 - SRI Turismo

Fecha de elaboracion: 2026-03-21

## Dominio y corpus

Dominio seleccionado: turismo y viajes.
Fuentes: sitios de guias de viaje y resenas turisticas (por ejemplo, Wikivoyage).

El corpus inicial se almacena en `data/raw/` y se procesa con el pipeline de preprocesamiento para generar tokens normalizados en `data/processed/`.

## Modelo de recuperacion (LSI)

Se utiliza Latent Semantic Indexing (LSI) sobre la matriz TF-IDF (documentos x terminos).

Resumen matematico:

- Matriz TF-IDF: \(A \in \mathbb{R}^{D \times T}\)
- Descomposicion: \(A \approx U_k \Sigma_k V_k^T\)
- Vectores de documentos: \(U_k \Sigma_k\)
- Proyeccion de consulta: \(q_k = q \, V_k^T\)
- Similaridad: coseno entre \(q_k\) y los vectores de documentos

Artefactos persistidos:

- `data/index/lsi_model.pkl`
- `data/index/doc_vectors.npy`

## Motor de busqueda

Flujo basico:

1. Preprocesar consulta (limpieza, tokenizacion, stopwords, stemming).
2. Vectorizar con TF-IDF usando el mismo vocabulario.
3. Proyectar al espacio LSI.
4. Calcular similitud coseno y ordenar resultados (Top-k).

## Estadisticas basicas del corpus

Calculadas sobre `data/raw/crawl/structured/20260315_104755/documents.jsonl`:

- Documentos: 100
- Total de palabras: 56164
- Promedio de palabras por documento: 561.64
- Minimo: 0
- Maximo: 11383

## Bibliografia

- Deerwester, S., Dumais, S., Furnas, G., Landauer, T., Harshman, R. (1990). Indexing by Latent Semantic Analysis.
- Manning, C. D., Raghavan, P., Schutze, H. (2008). Introduction to Information Retrieval.
- Documentacion de scikit-learn: `TruncatedSVD` y similitud coseno.

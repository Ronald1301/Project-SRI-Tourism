# Refinamiento de la Base de Datos Vectorial (FAISS + HNSW + Chunking)

Este documento describe las mejoras implementadas en `src/vector_db/vector_store.py` y su justificación técnica, alineadas con los criterios de: búsqueda más rápida/estable y persistencia correcta.

## 1. Problemas de la implementación original

La versión previa presentaba estas limitaciones estructurales:

1. Búsqueda `top-k` basada en operaciones globales sobre todo el corpus.
   - Similitud calculada contra todos los vectores.
   - Selección por ordenamiento completo (`argsort`) o equivalente.
   - Escalado pobre para corpus grandes.

2. Modelo de metadatos acoplado por posición.
   - `metadata: list[dict]` obliga sincronía estricta por índice.
   - Acceso por `doc_id` no directo.
   - Mayor fragilidad en actualizaciones incrementales.

3. Granularidad semántica insuficiente.
   - 1 documento = 1 embedding.
   - Documentos largos mezclan múltiples tópicos en un solo vector, degradando recall.

4. Persistencia incompleta del motor de búsqueda.
   - Se persistían vectores, pero no necesariamente un índice ANN listo para consultar.
   - Mayor costo de reconstrucción al reiniciar.

5. Flujo de actualización poco incremental.
   - Tendencia a reconstrucción completa ante nueva data.

---

## 2. Optimización del almacenamiento

Se diseñó una persistencia por artefactos desacoplados:

1. `embeddings.npy`
   - Embeddings de chunks en `float32`.
   - Carga con `mmap_mode="r"` para reducir carga de RAM.

2. `faiss.index`
   - Índice FAISS persistido (`faiss.write_index`) y recargado (`faiss.read_index`).

3. `doc_id_to_meta.json`
   - Diccionario `doc_id -> metadata` para acceso O(1) promedio.

4. `index_to_doc_id.json`
   - Vector posicional `chunk_index -> doc_id` para mapear hits de índice a entidad documental.

5. `meta.json`
   - Manifiesto de versión/formato/configuración:
     - `faiss_metric`
     - `faiss_index_type`
     - `hnsw_m`
     - `hnsw_ef_construction`
     - `hnsw_ef_search`
     - chunking (`chunk_size`, `chunk_overlap`)

Beneficio: separación clara entre plano vectorial, índice ANN y capa semántica de metadatos.

---

## 3. Uso de FAISS

### Implementación actual

1. Índice principal: `IndexHNSWFlat`.
   - Tipo por defecto en preset: `FAISS_INDEX_TYPE = "hnsw"`.
   - Hiperparámetros expuestos:
     - `HNSW_M`
     - `HNSW_EF_CONSTRUCTION`
     - `HNSW_EF_SEARCH`
   - Justificación estructural:
     - HNSW organiza los embeddings como un grafo de proximidad multinivel.
     - La búsqueda navega regiones prometedoras del espacio vectorial.
     - Evita comparar exhaustivamente contra todos los vectores, a diferencia de Flat.

2. Soporte de fallback exacto:
   - También se permite `flat` (`IndexFlatIP` / `IndexFlatL2`) para comparación y pruebas.

3. Métricas soportadas:
   - `ip` (inner product)
   - `l2`
   - Definiciones y fórmulas:
     - `ip` (Inner Product / producto interno):
       - Para dos vectores `x, y` en `R^d`:
       - `IP(x, y) = sum(i=1..d) [x_i * y_i]`
       - En FAISS con `IndexFlatIP` o HNSW métrica IP, un valor mayor implica mayor similitud.
     - `l2` (distancia euclidiana):
       - `||x - y||_2 = sqrt( sum(i=1..d) [(x_i - y_i)^2] )`
       - En FAISS normalmente se trabaja con distancia L2 al cuadrado:
       - `||x - y||_2^2 = sum(i=1..d) [(x_i - y_i)^2]`
       - En L2, un valor menor implica mayor similitud.
   - Relación con coseno:
     - Si se normalizan los vectores (`||x||_2 = ||y||_2 = 1`), entonces:
       - `cos(x, y) = x · y = IP(x, y)`
     - Por eso en esta implementación, con `normalize_embeddings=True`, usar `ip` es una forma eficiente de aproximar similitud coseno.

4. Normalización L2:
   - Cuando `normalize_embeddings=True`, base y query se normalizan con `faiss.normalize_L2`.
   - Con métrica `ip`, eso aproxima cosine similarity de manera eficiente.
   - ¿Qué hace exactamente `normalize_L2`?
     - Convierte cada embedding `x` en un vector unitario:
       - `x_hat = x / ||x||_2`, con `||x||_2 = sqrt( sum(i=1..d) [x_i^2] )`
     - Después de normalizar, todos los vectores tienen norma 1 (`||x_hat||_2 = 1`).
     - Efecto práctico:
       - Se reduce el impacto de la magnitud del vector.
       - La comparación se centra en la dirección semántica.
     - Consecuencia clave:
       - Para vectores normalizados, el producto interno coincide con coseno:
       - `x_hat · y_hat = cos(theta)`
       - Por eso `ip` + `normalize_L2` es una combinación estándar en recuperación semántica.

---

## 4. Complejidad temporal

### Antes

1. Similitud contra `N` vectores: `O(N * d)`.
2. Ranking final por ordenamiento global: típico `O(N log N)`.

### Ahora (HNSW)

1. Consulta ANN con HNSW:
   - Búsqueda aproximada sublineal en promedio (dependiente del grafo y `efSearch`).
   - Mejor latencia práctica que escaneo exhaustivo en corpus medianos/grandes.

2. Agregación chunk -> documento:
   - Se procesa solo el pool de candidatos (`chunk_k = top_k * factor`).
   - Costo adicional acotado por ese subconjunto.

3. Tradeoff controlable:
   - Mayor `efSearch` => mejor recall, mayor costo.
   - Menor `efSearch` => más velocidad, menor recall.
   - Costos de HNSW frente a Flat:
     - Mayor consumo de memoria por la estructura de grafo (vecindades por nodo).
     - Construcción del índice más costosa (tiempo de build más alto).
     - Resultados aproximados en lugar de exactos.
   - Justificación del tradeoff:
     - En recuperación semántica a escala, el aumento de latencia y costo de Flat suele ser más crítico.
     - Se prioriza HNSW porque mejora significativamente latencia y escalabilidad con calidad controlable por parámetros.

Nota: si se usa `flat`, la búsqueda vuelve a ser exhaustiva (exacta), útil para benchmarking.

---

## 5. Persistencia

Persistencia implementada end-to-end:

1. Índice FAISS:
   - Escritura: `faiss.write_index(...)`
   - Lectura: `faiss.read_index(...)`

2. Embeddings:
   - Persistidos en `.npy`.
   - Cargados con memory mapping (`np.load(..., mmap_mode="r")`).
   - Uso de `float32`:
     - Es el estándar de facto en sistemas vectoriales modernos.
     - Reduce memoria/IO frente a `float64` sin degradación relevante para retrieval.
     - Los embeddings de modelos transformer son suficientemente robustos numéricamente para búsqueda semántica en `float32`.

3. Mapeos documentales:
   - `doc_id_to_meta.json`
   - `index_to_doc_id.json`

4. Recuperación resiliente:
   - Si falta `faiss.index`, el sistema puede reconstruir el índice desde embeddings + configuración en `meta.json`.
   - Incluye compatibilidad hacia atrás con artefactos antiguos (`doc_ids`/`metadata` en `meta.json`).

5. Validación de integridad:
   - Se comprueba que `embeddings.shape[0] == len(index_to_doc_id)`.
   - Se garantiza metadata mínima por `doc_id`.

---

## 6. Estrategias de similitud

La recuperación se realiza en dos niveles:

1. Nivel chunk:
   - Se buscan chunks relevantes en FAISS.

2. Nivel documento:
   - Se agrupan chunks por `doc_id`.
   - Se usa mejor score por documento (`max pooling` de chunks).
   - Se expone `chunk_hits` para trazabilidad.
   - Sobre-recuperación con `chunk_pool_factor`:
     - La búsqueda inicial usa `chunk_k = top_k * factor`.
     - Motivo: varios chunks top pueden pertenecer al mismo documento.
     - Sin sobre-recuperación, se reduce diversidad documental y puede perderse recall a nivel documento final.

Ventaja: mejor comportamiento semántico en documentos largos, donde la relevancia suele estar localizada.

---

## 7. Escalabilidad

Mejoras aplicadas para crecimiento progresivo:

1. Ingesta incremental (`add_documents(...)`):
   - Genera embeddings solo de nuevos documentos.
   - `index.add(...)` directo al índice ANN.
   - Actualiza `index_to_doc_id` y `doc_id_to_meta`.
   - Permite persistencia incremental.

2. Diseño desacoplado de estado:
   - Índice ANN, embeddings y metadatos evolucionan de forma controlada.

3. Ruta de evolución:
   - La arquitectura permite migrar a otros índices FAISS (IVF/PQ/HNSW híbrido) si el tamaño crece aún más.

Observación operativa:
- `add_documents(...)` mantiene simplicidad y consistencia, pero actualmente reescribe artefactos al persistir. Es incremental en lógica de indexación, no un append binario in-place de todos los archivos.

---

## 8. Justificación técnica

1. Rendimiento:
   - FAISS HNSW reduce latencia de consulta frente a enfoques exhaustivos puros.
   - `chunk_pool_factor` permite controlar recall/costo en agregación.

2. Robustez:
   - Persistencia explícita de índice + embeddings + mapeos evita estados ambiguos.
   - Validaciones internas reducen errores silenciosos.

3. Calidad de recuperación:
   - Chunking + agregación por documento mejora recall semántico en textos largos.
   - Normalización L2 + `ip` da una aproximación efectiva a coseno.

4. Mantenibilidad:
   - `doc_id_to_meta` simplifica lookup y actualización por documento.
   - `index_to_doc_id` formaliza el puente chunk/documento.

5. Escalabilidad:
   - La API incremental reduce necesidad de rebuild total.
   - Parámetros HNSW expuestos en preset facilitan tuning por entorno.

---

## Conclusión

La base vectorial queda en una arquitectura más sólida para producción académica:

1. motor ANN (HNSW) configurable,
2. persistencia consistente del índice FAISS,
3. carga eficiente de embeddings con memory mapping,
4. estructura de metadatos orientada a acceso directo,
5. ingesta incremental con `add_documents(...)`,
6. chunking documental para mejor calidad semántica.

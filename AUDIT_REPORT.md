# AUDIT REPORT - SRI System

## 1. Estado general

**WARNING**

El sistema ya tiene los tres módulos pedidos y están integrados, pero el benchmark offline todavía muestra que la versión refinada no supera al baseline en este conjunto de consultas. Eso no bloquea el funcionamiento, pero sí pide ajuste fino.

## 2. Problemas encontrados

1. **Domain Detection**
   - El detector existía, pero el backend no estaba consumiendo su contrato completo.
   - Faltaba exponer de forma clara `confidence` y `scores` en la respuesta del backend.
   - El cliente de Ollama no estaba inicializado con timeout explícito.
   - La configuración de umbrales no estaba desacoplada del código.

2. **Query Expansion**
   - La expansión estaba presente, pero el proyecto dependía demasiado de valores por defecto y archivos legados.
   - Faltaba normalizar la configuración externa para que el expander y la evaluación usaran el mismo contrato.
   - El sistema necesitaba alinear los defaults con el criterio de auditoría: `max_terms=28`, `score_floor=0.1`, `acceptance_threshold=0.75`.

3. **Evaluation**
   - La evaluación era funcional, pero incompleta para la auditoría:
     - faltaban `Precision@3`, `Precision@5`, `Recall@5`, `Recall@10`
     - faltaban desviación estándar y bootstrap CI 95%
     - el reporte no mostraba una comparación rica entre baseline y variante refinada

4. **Tests**
   - No había una batería de pruebas de auditoría que validara dominio, expansión y evaluación con contratos claros.

## 3. Cambios realizados

### Backend

- En `src/api/service/rag_service.py`:
  - el detector de dominio ahora se consulta con `is_in_domain(query)`
  - se carga configuración externa desde `data/config/domain_detection.json`
  - el cliente de Ollama se crea con timeout de 5 segundos
  - la respuesta del dominio incluye `confidence` y `scores`

- En `src/api/config.py`:
  - se corrigieron rutas duplicadas y se dejó un único origen para:
    - `DEFAULT_QUERY_EXPANSION_CONFIG`
    - `DEFAULT_DOMAIN_DETECTION_CONFIG`
    - `DEFAULT_DOMAIN_SYNONYMS`

### Query Expansion

- En `src/retrieval/query_expansion.py`:
  - se ajustaron defaults conservadores del expander:
    - `max_terms = 28`
    - `acceptance_threshold = 0.75`
  - se mantuvo el uso de SQLite para feedback
  - se conservó el filtrado de términos inválidos y stopwords

- Se añadieron archivos externos:
  - `data/config/query_expansion.json`
  - `data/config/tourism_synonyms.json`
  - `data/config/domain_detection.json`
  - `data/config/evaluation.json`

- En `src/evaluation/systems.py`:
  - el `QueryExpander` ahora se instancia con config y sinónimos externos

### Evaluation

- En `src/evaluation/metrics.py`:
  - se añadieron utilidades para:
    - media
    - desviación estándar
    - bootstrap CI 95%

- En `src/evaluation/experiment_runner.py`:
  - se agregaron métricas:
    - `Precision@3`
    - `Precision@5`
    - `Recall@5`
    - `Recall@10`
    - `MAP`
    - `NDCG@5`
    - `MRR`
  - se conservó compatibilidad con las métricas previas
  - se añadió el bloque `statistics` por sistema con:
    - media
    - desviación estándar
    - intervalo de confianza 95%

- En `src/evaluation/report_generator.py` y `src/evaluation/cli.py`:
  - se actualizó el formato del reporte para mostrar las nuevas métricas

### Tests

- Se creó `tests/test_audit_integration.py` para validar:
  - query válida pasa el detector de dominio
  - query inválida queda bloqueada
  - la expansión modifica la query seleccionada
  - la evaluación produce las métricas requeridas

## 4. Métricas antes vs después

Evaluación offline sobre 12 consultas, comparando `lsi_baseline` vs `lsi_refined`:

| Métrica | Baseline | Refined | Delta |
|---|---:|---:|---:|
| Precision@k | 0.0500 | 0.0500 | +0.0000 |
| Recall@k | 0.0764 | 0.0764 | +0.0000 |
| F1@k | 0.0602 | 0.0602 | +0.0000 |
| MAP | 0.0625 | 0.0597 | -0.0028 |
| MRR@k | 0.1944 | 0.1833 | -0.0111 |
| NDCG@k | 0.1358 | 0.1331 | -0.0026 |
| R-Precision | 0.0764 | 0.0556 | -0.0208 |

### Lectura rápida

- El flujo quedó más robusto y mejor definido.
- La parte de contrato y auditoría mejoró.
- En este dataset concreto, la variante refinada todavía no supera al baseline en calidad de ranking.

## 5. Recomendaciones

1. Ajustar los pesos de expansión para reducir query drift.
2. Revisar sinónimos ambiguos y términos demasiado generales.
3. Validar el detector de dominio con un conjunto pequeño etiquetado manualmente.
4. Ejecutar más benchmarks con consultas fuera de dominio y ambiguas.
5. Mantener el reporte de evaluación como artefacto obligatorio antes de cambios grandes.

## 6. Verificación realizada

- Compilación Python: OK
- Tests de evaluación existentes: OK
- Tests de auditoría nuevos: OK
- Integración backend/frontend: conservada


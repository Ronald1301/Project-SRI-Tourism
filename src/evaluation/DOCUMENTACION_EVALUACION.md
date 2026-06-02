# Documentacion del modulo de evaluacion

Este modulo evalua el comportamiento de los sistemas de recuperacion de informacion del proyecto usando qrels y metricas clasicas de IR.

## Proposito

La evaluacion offline permite comparar de forma objetiva distintas variantes del sistema:

- `lsi_baseline`
- `lsi_refined`
- `lsi_expanded`
- `vectorial`
- `hybrid_search`

La evaluacion no mide la generacion final del RAG, sino la calidad de la recuperacion de documentos relevantes.

## Por que a veces solo aparece baseline vs refined

Existe una funcion de compatibilidad hacia atras:

- `evaluate_searcher(...)`

Esa funcion esta pensada para mantener soporte con comandos antiguos y evalua solo dos sistemas:

- `lsi_baseline`
- `lsi_refined`

Si se quiere comparar todos los sistemas disponibles, debe usarse la evaluacion general:

- `evaluate_systems(...)`

o la CLI del modulo.

## Sistemas soportados

Los sistemas registrados actualmente son:

- `lsi_baseline`: TF-IDF + LSI sin reranking
- `lsi_refined`: TF-IDF + LSI + reranking
- `lsi_expanded`: TF-IDF + LSI + expansion + reranking
- `vectorial`: embeddings vectoriales
- `hybrid_search`: fusion hibrida RRF

## Metricas calculadas

El modulo calcula:

- `Precision@3`
- `Precision@5`
- `Recall@5`
- `Recall@10`
- `MAP`
- `NDCG@5`
- `MRR`
- `Precision@k`
- `Recall@k`
- `F1@k`
- `AP`
- `MRR@k`
- `NDCG@k`
- `R-Precision`

Adicionalmente, el reporte incluye:

- media
- desviacion estandar
- intervalo de confianza bootstrap al 95%
- delta contra baseline

## Como ejecutar la evaluacion

### Comparar todos los sistemas

```powershell
.\venv\Scripts\python.exe -m src.evaluation.cli --systems all --top-k 5
```

### Comparar sistemas especificos

```powershell
.\venv\Scripts\python.exe -m src.evaluation.cli --systems lsi_baseline,lsi_refined,lsi_expanded,vectorial,hybrid_search --top-k 5
```

### Usar un archivo de qrels distinto

```powershell
.\venv\Scripts\python.exe -m src.evaluation.cli --qrels data/evaluation/rec01_qrels.json --top-k 5
```

## Archivos de salida

Por defecto el modulo escribe:

- `data/evaluation/results/eval_report.json`
- `data/evaluation/results/eval_report.md`

Cuando se usa el comparativo ampliado, tambien puede generar:

- `data/evaluation/results/audit_comparison.json`
- `data/evaluation/results/audit_comparison.md`

## Flujo interno

1. Se cargan los qrels.
2. Se construyen los sistemas solicitados.
3. Se ejecuta cada sistema sobre cada consulta.
4. Se extraen los `doc_id` recuperados.
5. Se calculan las metricas por consulta.
6. Se resume el resultado por sistema.
7. Se comparan los sistemas contra el baseline.
8. Se genera el reporte JSON y Markdown.

## Relacion con el resto del proyecto

La evaluacion se conecta con:

- `src/retrieval/search.py`
- `src/retrieval/query_expansion.py`
- `src/RAG/rag_pipeline.py`
- `src/evaluation/metrics.py`
- `src/evaluation/systems.py`
- `src/evaluation/experiment_runner.py`
- `src/evaluation/report_generator.py`

## Nota importante

Si se usa `evaluate_searcher(...)`, la comparacion sera solo entre:

- `lsi_baseline`
- `lsi_refined`

Eso no es un error: es una funcion legacy de compatibilidad. Para evaluar todo el conjunto de sistemas, usar `evaluate_systems(...)` o la CLI.


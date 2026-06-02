# Reporte de evaluacion IR

- Consultas evaluadas: 12
- Top-k: 5
- Baseline: `lsi_baseline`

## Resumen por sistema

| Sistema | Estado | P@3 | P@5 | R@5 | R@10 | MAP | NDCG@5 | MRR | P@k | R@k | F1@k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lsi_baseline | ok | 0.1389 | 0.1000 | 0.1278 | 0.2944 | 0.0993 | 0.1353 | 0.2891 | 0.1000 | 0.1278 | 0.1105 |
| lsi_refined | ok | 0.1667 | 0.1000 | 0.1278 | 0.3778 | 0.0993 | 0.1295 | 0.2821 | 0.1000 | 0.1278 | 0.1105 |
| lsi_expanded | ok | 0.1111 | 0.0833 | 0.1000 | 0.1486 | 0.0670 | 0.0971 | 0.2389 | 0.0833 | 0.1000 | 0.0897 |
| vectorial | ok | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hybrid_search | ok | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Delta contra baseline

| Sistema | Delta F1@k | Delta NDCG@k | Delta MRR@k |
|---|---:|---:|---:|
| lsi_refined | +0.0000 | -0.0058 | -0.0208 |
| lsi_expanded | -0.0208 | -0.0382 | -0.0319 |
| vectorial | -0.1105 | -0.1353 | -0.2569 |
| hybrid_search | -0.1105 | -0.1353 | -0.2569 |

## Comparacion con metodos normales

Baseline de referencia: `lsi_baseline`

| Sistema | MAP | NDCG@5 | MRR | F1@k | Delta MAP | Delta NDCG@5 | Delta MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| lsi_baseline | 0.0993 | 0.1353 | 0.2891 | 0.1105 | +0.0000 | +0.0000 | +0.0000 |
| lsi_refined | 0.0993 | 0.1295 | 0.2821 | 0.1105 | +0.0000 | -0.0058 | -0.0069 |
| lsi_expanded | 0.0670 | 0.0971 | 0.2389 | 0.0897 | -0.0323 | -0.0382 | -0.0502 |
| vectorial | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0993 | -0.1353 | -0.2891 |
| hybrid_search | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0993 | -0.1353 | -0.2891 |

El baseline `lsi_baseline` mantiene la referencia de comparación para los metodos normales.

## Estadisticas por sistema

### lsi_baseline

| Metricas | Media | Desv. Std. | CI 95% |
|---|---:|---:|---:|
| precision_at_3 | 0.1389 | 0.1643 | [0.0556, 0.2222] |
| precision_at_5 | 0.1000 | 0.1000 | [0.0500, 0.1500] |
| recall_at_5 | 0.1278 | 0.1351 | [0.0583, 0.2083] |
| recall_at_10 | 0.2944 | 0.2040 | [0.1806, 0.4056] |
| map | 0.0993 | 0.0956 | [0.0511, 0.1566] |
| ndcg_at_5 | 0.1353 | 0.1576 | [0.0562, 0.2338] |
| mrr | 0.2891 | 0.2845 | [0.1482, 0.4593] |
| precision_at_k | 0.1000 | 0.1000 | [0.0500, 0.1500] |
| recall_at_k | 0.1278 | 0.1351 | [0.0583, 0.2083] |
| f1_at_k | 0.1105 | 0.1119 | [0.0527, 0.1730] |
| ap | 0.0993 | 0.0956 | [0.0511, 0.1566] |
| mrr_at_k | 0.2569 | 0.3071 | [0.1042, 0.4444] |
| ndcg_at_k | 0.1353 | 0.1576 | [0.0562, 0.2338] |
| r_precision | 0.1278 | 0.1351 | [0.0583, 0.2083] |

### lsi_refined

| Metricas | Media | Desv. Std. | CI 95% |
|---|---:|---:|---:|
| precision_at_3 | 0.1667 | 0.1667 | [0.0833, 0.2500] |
| precision_at_5 | 0.1000 | 0.1000 | [0.0500, 0.1500] |
| recall_at_5 | 0.1278 | 0.1351 | [0.0583, 0.2083] |
| recall_at_10 | 0.3778 | 0.2626 | [0.2278, 0.5306] |
| map | 0.0993 | 0.0613 | [0.0658, 0.1326] |
| ndcg_at_5 | 0.1295 | 0.1466 | [0.0569, 0.2146] |
| mrr | 0.2821 | 0.2009 | [0.1758, 0.3897] |
| precision_at_k | 0.1000 | 0.1000 | [0.0500, 0.1500] |
| recall_at_k | 0.1278 | 0.1351 | [0.0583, 0.2083] |
| f1_at_k | 0.1105 | 0.1119 | [0.0527, 0.1730] |
| ap | 0.0993 | 0.0613 | [0.0658, 0.1326] |
| mrr_at_k | 0.2361 | 0.2402 | [0.1111, 0.3750] |
| ndcg_at_k | 0.1295 | 0.1466 | [0.0569, 0.2146] |
| r_precision | 0.1278 | 0.1351 | [0.0583, 0.2083] |

### lsi_expanded

| Metricas | Media | Desv. Std. | CI 95% |
|---|---:|---:|---:|
| precision_at_3 | 0.1111 | 0.1571 | [0.0278, 0.1944] |
| precision_at_5 | 0.0833 | 0.0986 | [0.0333, 0.1333] |
| recall_at_5 | 0.1000 | 0.1238 | [0.0306, 0.1694] |
| recall_at_10 | 0.1486 | 0.1677 | [0.0625, 0.2431] |
| map | 0.0670 | 0.0873 | [0.0208, 0.1179] |
| ndcg_at_5 | 0.0971 | 0.1208 | [0.0288, 0.1638] |
| mrr | 0.2389 | 0.3073 | [0.0833, 0.4306] |
| precision_at_k | 0.0833 | 0.0986 | [0.0333, 0.1333] |
| recall_at_k | 0.1000 | 0.1238 | [0.0306, 0.1694] |
| f1_at_k | 0.0897 | 0.1072 | [0.0318, 0.1468] |
| ap | 0.0670 | 0.0873 | [0.0208, 0.1179] |
| mrr_at_k | 0.2250 | 0.3139 | [0.0583, 0.4167] |
| ndcg_at_k | 0.0971 | 0.1208 | [0.0288, 0.1638] |
| r_precision | 0.1000 | 0.1238 | [0.0306, 0.1694] |

### vectorial

| Metricas | Media | Desv. Std. | CI 95% |
|---|---:|---:|---:|
| precision_at_3 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| precision_at_5 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| recall_at_5 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| recall_at_10 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| map | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| ndcg_at_5 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| mrr | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| precision_at_k | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| recall_at_k | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| f1_at_k | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| ap | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| mrr_at_k | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| ndcg_at_k | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| r_precision | 0.0000 | 0.0000 | [0.0000, 0.0000] |

### hybrid_search

| Metricas | Media | Desv. Std. | CI 95% |
|---|---:|---:|---:|
| precision_at_3 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| precision_at_5 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| recall_at_5 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| recall_at_10 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| map | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| ndcg_at_5 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| mrr | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| precision_at_k | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| recall_at_k | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| f1_at_k | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| ap | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| mrr_at_k | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| ndcg_at_k | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| r_precision | 0.0000 | 0.0000 | [0.0000, 0.0000] |


## Analisis cuantitativo

- El mejor compromiso Precision/Recall segun F1@k lo obtiene `lsi_baseline` con 0.1105.
- El mejor ordenamiento global segun NDCG@k lo obtiene `lsi_baseline` con 0.1353.
- La mejor ubicacion temprana del primer relevante segun MRR@k la obtiene `lsi_baseline` con 0.2569.
- Ningun sistema evaluado supera al baseline en F1@k con este conjunto de consultas.

## Advertencias de validacion

- hybrid_search/q10: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- hybrid_search/q10: sin resultados.
- hybrid_search/q11: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- hybrid_search/q11: sin resultados.
- hybrid_search/q12: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- hybrid_search/q12: sin resultados.
- hybrid_search/q1: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- hybrid_search/q1: sin resultados.
- hybrid_search/q2: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- hybrid_search/q2: sin resultados.
- hybrid_search/q3: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- hybrid_search/q3: sin resultados.
- hybrid_search/q4: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- hybrid_search/q4: sin resultados.
- hybrid_search/q5: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- hybrid_search/q5: sin resultados.
- hybrid_search/q6: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- hybrid_search/q6: sin resultados.
- hybrid_search/q7: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- hybrid_search/q7: sin resultados.
- hybrid_search/q8: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- hybrid_search/q8: sin resultados.
- hybrid_search/q9: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- hybrid_search/q9: sin resultados.
- vectorial/q10: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- vectorial/q10: sin resultados.
- vectorial/q11: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- vectorial/q11: sin resultados.
- vectorial/q12: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- vectorial/q12: sin resultados.
- vectorial/q1: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- vectorial/q1: sin resultados.
- vectorial/q2: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- vectorial/q2: sin resultados.
- vectorial/q3: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- vectorial/q3: sin resultados.
- vectorial/q4: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- vectorial/q4: sin resultados.
- vectorial/q5: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- vectorial/q5: sin resultados.
- vectorial/q6: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- vectorial/q6: sin resultados.
- vectorial/q7: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- vectorial/q7: sin resultados.
- vectorial/q8: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- vectorial/q8: sin resultados.
- vectorial/q9: error recuperando resultados: RAGPipeline.retrieve() got an unexpected keyword argument 'search_mode'
- vectorial/q9: sin resultados.

## Mejores y peores consultas por sistema

- `lsi_baseline` mejor: q7 (alojamiento en varadero) F1@k=0.2500; peor: q3 (luna de miel en cuba) F1@k=0.0000.
- `lsi_refined` mejor: q7 (alojamiento en varadero) F1@k=0.2500; peor: q3 (luna de miel en cuba) F1@k=0.0000.
- `lsi_expanded` mejor: q7 (alojamiento en varadero) F1@k=0.2500; peor: q3 (luna de miel en cuba) F1@k=0.0000.
- `vectorial` mejor: q1 (que ver en la habana vieja) F1@k=0.0000; peor: q1 (que ver en la habana vieja) F1@k=0.0000.
- `hybrid_search` mejor: q1 (que ver en la habana vieja) F1@k=0.0000; peor: q1 (que ver en la habana vieja) F1@k=0.0000.

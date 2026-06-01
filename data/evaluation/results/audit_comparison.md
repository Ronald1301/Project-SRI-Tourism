# Reporte de evaluacion IR

- Consultas evaluadas: 12
- Top-k: 5
- Baseline: `lsi_baseline`

## Resumen por sistema

| Sistema | Estado | P@3 | P@5 | R@5 | R@10 | MAP | NDCG@5 | MRR | P@k | R@k | F1@k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lsi_baseline | ok | 0.0833 | 0.0500 | 0.0764 | 0.0764 | 0.0625 | 0.1358 | 0.1944 | 0.0500 | 0.0764 | 0.0602 |
| lsi_refined | ok | 0.0556 | 0.0500 | 0.0764 | 0.0764 | 0.0597 | 0.1331 | 0.1833 | 0.0500 | 0.0764 | 0.0602 |

## Delta contra baseline

| Sistema | Delta F1@k | Delta NDCG@k | Delta MRR@k |
|---|---:|---:|---:|
| lsi_refined | +0.0000 | -0.0026 | -0.0111 |

## Comparacion con metodos normales

Baseline de referencia: `lsi_baseline`

| Sistema | MAP | NDCG@5 | MRR | F1@k | Delta MAP | Delta NDCG@5 | Delta MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| lsi_baseline | 0.0625 | 0.1358 | 0.1944 | 0.0602 | +0.0000 | +0.0000 | +0.0000 |
| lsi_refined | 0.0597 | 0.1331 | 0.1833 | 0.0602 | -0.0028 | -0.0026 | -0.0111 |

El baseline `lsi_baseline` mantiene la referencia de comparación para los metodos normales.

## Estadisticas por sistema

### lsi_baseline

| Metricas | Media | Desv. Std. | CI 95% |
|---|---:|---:|---:|
| precision_at_3 | 0.0833 | 0.1443 | [0.0000, 0.1667] |
| precision_at_5 | 0.0500 | 0.0866 | [0.0000, 0.1000] |
| recall_at_5 | 0.0764 | 0.1338 | [0.0000, 0.1597] |
| recall_at_10 | 0.0764 | 0.1338 | [0.0000, 0.1597] |
| map | 0.0625 | 0.1233 | [0.0000, 0.1458] |
| ndcg_at_5 | 0.1358 | 0.2752 | [0.0000, 0.3221] |
| mrr | 0.1944 | 0.3716 | [0.0000, 0.4444] |
| precision_at_k | 0.0500 | 0.0866 | [0.0000, 0.1000] |
| recall_at_k | 0.0764 | 0.1338 | [0.0000, 0.1597] |
| f1_at_k | 0.0602 | 0.1044 | [0.0000, 0.1227] |
| ap | 0.0625 | 0.1233 | [0.0000, 0.1458] |
| mrr_at_k | 0.1944 | 0.3716 | [0.0000, 0.4444] |
| ndcg_at_k | 0.1358 | 0.2752 | [0.0000, 0.3221] |
| r_precision | 0.0764 | 0.1338 | [0.0000, 0.1597] |

### lsi_refined

| Metricas | Media | Desv. Std. | CI 95% |
|---|---:|---:|---:|
| precision_at_3 | 0.0556 | 0.1242 | [0.0000, 0.1389] |
| precision_at_5 | 0.0500 | 0.0866 | [0.0000, 0.1000] |
| recall_at_5 | 0.0764 | 0.1338 | [0.0000, 0.1597] |
| recall_at_10 | 0.0764 | 0.1338 | [0.0000, 0.1597] |
| map | 0.0597 | 0.1231 | [0.0000, 0.1431] |
| ndcg_at_5 | 0.1331 | 0.2753 | [0.0000, 0.3195] |
| mrr | 0.1833 | 0.3693 | [0.0000, 0.4333] |
| precision_at_k | 0.0500 | 0.0866 | [0.0000, 0.1000] |
| recall_at_k | 0.0764 | 0.1338 | [0.0000, 0.1597] |
| f1_at_k | 0.0602 | 0.1044 | [0.0000, 0.1227] |
| ap | 0.0597 | 0.1231 | [0.0000, 0.1431] |
| mrr_at_k | 0.1833 | 0.3693 | [0.0000, 0.4333] |
| ndcg_at_k | 0.1331 | 0.2753 | [0.0000, 0.3195] |
| r_precision | 0.0556 | 0.1242 | [0.0000, 0.1389] |


## Analisis cuantitativo

- El mejor compromiso Precision/Recall segun F1@k lo obtiene `lsi_baseline` con 0.0602.
- El mejor ordenamiento global segun NDCG@k lo obtiene `lsi_baseline` con 0.1358.
- La mejor ubicacion temprana del primer relevante segun MRR@k la obtiene `lsi_baseline` con 0.1944.
- Ningun sistema evaluado supera al baseline en F1@k con este conjunto de consultas.

## Mejores y peores consultas por sistema

- `lsi_baseline` mejor: q10 (transporte turistico en varadero) F1@k=0.2500; peor: q1 (que ver en la habana vieja) F1@k=0.0000.
- `lsi_refined` mejor: q10 (transporte turistico en varadero) F1@k=0.2500; peor: q1 (que ver en la habana vieja) F1@k=0.0000.

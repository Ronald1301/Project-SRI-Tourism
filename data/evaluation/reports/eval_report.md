# Reporte de evaluacion IR

- Consultas evaluadas: 12
- Top-k: 5
- Baseline: `lsi_baseline`

## Resumen por sistema

| Sistema | Estado | P@k | Recall@k | F1@k | MAP | MRR@k | NDCG@k | R-Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lsi_baseline | ok | 0.0500 | 0.0764 | 0.0602 | 0.0625 | 0.1944 | 0.1358 | 0.0764 |
| lsi_refined | ok | 0.0500 | 0.0764 | 0.0602 | 0.0597 | 0.1833 | 0.1331 | 0.0556 |

## Delta contra baseline

| Sistema | Delta F1@k | Delta NDCG@k | Delta MRR@k |
|---|---:|---:|---:|
| lsi_refined | +0.0000 | -0.0026 | -0.0111 |

## Analisis cuantitativo

- El mejor compromiso Precision/Recall segun F1@k lo obtiene `lsi_baseline` con 0.0602.
- El mejor ordenamiento global segun NDCG@k lo obtiene `lsi_baseline` con 0.1358.
- La mejor ubicacion temprana del primer relevante segun MRR@k la obtiene `lsi_baseline` con 0.1944.
- Ningun sistema evaluado supera al baseline en F1@k con este conjunto de consultas.

## Mejores y peores consultas por sistema

- `lsi_baseline` mejor: q10 (transporte turistico en varadero) F1@k=0.2500; peor: q1 (que ver en la habana vieja) F1@k=0.0000.
- `lsi_refined` mejor: q10 (transporte turistico en varadero) F1@k=0.2500; peor: q1 (que ver en la habana vieja) F1@k=0.0000.

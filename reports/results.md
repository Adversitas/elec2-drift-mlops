# Retraining policies on Elec2

134 weekly batches of 336 half-hours; 4 warm-up weeks; 130 weeks scored prequentially.

| model | policy | mean_accuracy | p10_week_accuracy | worst_week_accuracy | promotions | model_fits | wall_seconds |
|---|---|---|---|---|---|---|---|
| hgb | static | 0.634 | 0.518 | 0.357 | 0 | 2 | 41.501 |
| hgb | periodic-4w | 0.764 | 0.651 | 0.491 | 32 | 34 | 312.163 |
| hgb | weekly | 0.776 | 0.675 | 0.515 | 130 | 132 | 1140.970 |
| hgb | weekly-expanding | 0.771 | 0.672 | 0.601 | 130 | 132 | 188.380 |
| hgb | psi-triggered | 0.776 | 0.675 | 0.515 | 128 | 258 | 238.264 |
| hgb | perf-triggered-5pt | 0.711 | 0.570 | 0.491 | 8 | 18 | 18.480 |
| hgb | perf-triggered-2pt | 0.720 | 0.570 | 0.491 | 13 | 28 | 28.833 |
| hgb | perf-triggered-2pt+gate | 0.740 | 0.633 | 0.500 | 13 | 29 | 27.555 |
| logreg | static | 0.697 | 0.567 | 0.402 | 0 | 2 | 0.528 |
| logreg | periodic-4w | 0.774 | 0.646 | 0.491 | 32 | 34 | 1.092 |
| logreg | weekly | 0.785 | 0.666 | 0.521 | 130 | 132 | 2.838 |
| logreg | weekly-expanding | 0.753 | 0.610 | 0.408 | 130 | 132 | 9.309 |
| logreg | psi-triggered | 0.785 | 0.666 | 0.521 | 128 | 258 | 5.031 |
| logreg | perf-triggered-5pt | 0.729 | 0.562 | 0.464 | 8 | 18 | 0.966 |
| logreg | perf-triggered-2pt | 0.729 | 0.562 | 0.464 | 12 | 26 | 1.052 |
| logreg | perf-triggered-2pt+gate | 0.757 | 0.626 | 0.470 | 10 | 25 | 1.066 |

Persistence baseline (previous half-hour's label, needs real-time labels): 0.853

Share of scored weeks where a feature's PSI vs the warm-up weeks exceeds 0.25: {"nswprice": 1.0, "nswdemand": 0.731, "vicprice": 0.638, "vicdemand": 0.638, "transfer": 0.638}

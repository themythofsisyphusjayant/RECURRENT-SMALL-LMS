# Recurrent Small Language Models

*Architecture, Training, and Evaluation — A Technical Report*

The full report is at
[report/Recurrent_Small_LMs_Technical_Report_NeurIPS.pdf](report/Recurrent_Small_LMs_Technical_Report_NeurIPS.pdf)
(22 pages, two-column; LaTeX sources in `report/`). This README states
the one executed result and what the report does and does not claim.

## Question

Do recurrent transformers gain anything from running extra loops at test
time?

## Answer

No. Perplexity falls from 29.4 to 4.4 as loops go from 1 to 4 (the trained
depth), then flatlines — 8, 12, and 16 loops buy nothing, and going past
the trained depth slightly degrades performance.

![Loop-depth sweep with 95% bootstrap CIs](figures/fig1_sweep_curve.png)

| Loops | 1 | 2 | **4 (trained)** | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| Perplexity | 29.38 | 11.18 | **4.41** | 4.90 | 4.91 | 4.91 |
| 95% CI | [28.30, 30.52] | [10.84, 11.54] | **[4.28, 4.55]** | [4.75, 5.06] | [4.76, 5.07] | [4.76, 5.07] |

The 4 → 8 degradation is real: paired per-sequence difference +0.106
loss, 95% CI [+0.101, +0.110]. Test-time loop scaling saturates at the
trained depth and slightly degrades beyond it.

![3D saturation surface: both metrics fall to loop 4, then flatline](figures/fig4_3d_saturation.png)

## The reference model

- **43,974,722 parameters** · dim 512 · GPT-2 BPE (50,257)
- Architecture: prelude (6 layers) → loop block ×4 (8 MoE experts, GQA
  attention) → coda (2 layers) → logits
- Trained on TinyStories, 120M tokens, loss 9.52 → 1.53; 4,850 steps on a
  Kaggle T4, October 2026

![44M architecture: prelude, loop block with MoE experts, coda](figures/fig2_architecture.png)

## Why GQA matters

At batch 8 / 2k context the multi-head attention KV cache (403 MB) is 4×
the weights — GQA reclaims roughly 300 MB. At heavy batching that is the
difference between fitting on one GPU or not.

![KV-cache heatmap across batch size and context length](figures/fig3_kv_heatmap.png)

## What is in the report

| Section | Topic | Status |
|---|---|---|
| 3 | Loop-depth extrapolation sweep (the executed study) | Measured, with bootstrap CIs |
| 4 | MoE routing dynamics, GQA efficiency, depth–width tradeoffs | Analytical |
| 5 | Curriculum protocol, loop-aware distillation | Proposed |
| 6 | Test-time compute taxonomy, quantization, mechanistic probes, TinyLoop-Bench | Mixed analytical / proposed |
| 7 | Conclusion and the key next experiment | — |

Proposed experiments are labelled as proposed. The only executed study is
the loop sweep. The confidence intervals capture sampling noise over
sequences; run-to-run seed variance is named in the report as the next
experiment, not measured here.

## Repo layout

```
├── report/     → the 22-page two-column PDF + full LaTeX sources
├── figures/    → all diagrams (PNG)
└── scripts/    → make_figures.py (regenerate every figure from the data)
```

Regenerate figures: `python3 scripts/make_figures.py` (needs matplotlib
+ numpy).

---

Jayant Dubey · October 2026. Built from a real training run — 4,850 steps
on a Kaggle T4, October 2026. README revised November 2026; results as
reported in October 2026, file audit 21 November 2026.

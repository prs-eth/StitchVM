# UnifiedReward-2.0 Qwen3.5 — Detailed results

[Guide](EXTENSION_UNIFIEDREWARD_QWEN35.md) · [README](../README.md)

VL-RewardBench evaluation of SD3.5 Medium stitched to UnifiedReward-2.0
Qwen3.5. Every run uses all 1,247 items; stitched runs use epoch-5 EMA weights.
The VAE encodes processor-native images rather than fixed 1024² inputs.
See the [evaluation protocol](EXTENSION_UNIFIEDREWARD_QWEN35.md#evaluation).

The official and explicit-preference parsers score the same generated answers.
Accuracy and macro averages are percentages. Verdict agreement uses items with
valid verdicts in both runs. McNemar p-values are exact, two-sided and
unadjusted for the 18 comparisons. Macro averages use the official script's
id-prefix categories and the dataset's source categories, respectively.

**4B, official parser**

| Run | Accuracy | Invalid | Macro, official-script categories | Macro, source categories | Verdict agreement | Correct only in original / stitched | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 75.30 | 49 | 73.50 | 69.18 | — | — | — |
| σ = 0 | 72.41 | 52 | 70.70 | 66.86 | 91.94 | 79 / 43 | 0.0014 |
| σ = 0.1 | 73.14 | 47 | 71.34 | 67.60 | 92.98 | 66 / 39 | 0.011 |
| σ = 0.25 | 72.41 | 48 | 70.75 | 66.92 | 92.18 | 79 / 43 | 0.0014 |
| σ = 0.5 | 72.09 | 41 | 70.11 | 66.44 | 89.12 | 104 / 64 | 0.0025 |
| σ = 0.75 | 70.73 | 54 | 68.39 | 66.25 | 82.56 | 152 / 95 | 0.00035 |
| σ = 0.9 | 69.69 | 85 | 67.79 | 65.65 | 79.65 | 184 / 114 | 6e-05 |

**4B, explicit-preference parser**

| Run | Accuracy | Unrecognized | Macro, official-script categories | Macro, source categories | Verdict agreement | Correct only in original / stitched | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 76.98 | 21 | 75.66 | 71.33 | — | — | — |
| σ = 0 | 74.42 | 22 | 73.13 | 69.18 | 92.04 | 69 / 37 | 0.0024 |
| σ = 0.1 | 74.90 | 21 | 73.55 | 69.79 | 93.11 | 58 / 32 | 0.008 |
| σ = 0.25 | 74.18 | 23 | 72.86 | 68.93 | 92.36 | 68 / 33 | 0.00064 |
| σ = 0.5 | 73.14 | 18 | 71.47 | 67.80 | 89.01 | 98 / 50 | 9.8e-05 |
| σ = 0.75 | 71.85 | 30 | 69.86 | 67.72 | 82.60 | 149 / 85 | 3.4e-05 |
| σ = 0.9 | 70.97 | 57 | 69.42 | 67.41 | 79.86 | 179 / 104 | 9.7e-06 |

**9B, official parser**

| Run | Accuracy | Invalid | Macro, official-script categories | Macro, source categories | Verdict agreement | Correct only in original / stitched | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 84.36 | 36 | 82.04 | 79.51 | — | — | — |
| σ = 0 | 82.92 | 41 | 80.37 | 77.86 | 96.74 | 37 / 19 | 0.022 |
| σ = 0.1 | 82.68 | 41 | 80.27 | 77.31 | 97.32 | 35 / 14 | 0.0038 |
| σ = 0.25 | 83.56 | 35 | 81.37 | 78.53 | 96.49 | 35 / 25 | 0.25 |
| σ = 0.5 | 81.96 | 39 | 79.30 | 76.80 | 93.69 | 66 / 36 | 0.0039 |
| σ = 0.75 | 79.47 | 42 | 76.65 | 74.64 | 88.77 | 113 / 52 | 2.3e-06 |
| σ = 0.9 | 79.31 | 44 | 76.86 | 73.89 | 88.21 | 119 / 56 | 2.2e-06 |

**9B, explicit-preference parser**

| Run | Accuracy | Unrecognized | Macro, official-script categories | Macro, source categories | Verdict agreement | Correct only in original / stitched | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 84.44 | 35 | 82.14 | 79.61 | — | — | — |
| σ = 0 | 83.08 | 38 | 80.58 | 78.07 | 96.75 | 34 / 17 | 0.024 |
| σ = 0.1 | 82.76 | 40 | 80.38 | 77.41 | 97.33 | 34 / 13 | 0.0031 |
| σ = 0.25 | 83.48 | 35 | 81.26 | 78.42 | 96.50 | 35 / 23 | 0.15 |
| σ = 0.5 | 82.04 | 37 | 79.40 | 76.91 | 93.62 | 65 / 35 | 0.0035 |
| σ = 0.75 | 79.63 | 39 | 76.86 | 74.85 | 88.80 | 111 / 51 | 2.8e-06 |
| σ = 0.9 | 79.55 | 36 | 77.22 | 74.27 | 88.20 | 116 / 55 | 3.6e-06 |

**27B, official parser**

| Run | Accuracy | Invalid | Macro, official-script categories | Macro, source categories | Verdict agreement | Correct only in original / stitched | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 80.91 | 55 | 78.74 | 74.85 | — | — | — |
| σ = 0 | 80.27 | 53 | 77.94 | 74.33 | 97.54 | 29 / 21 | 0.32 |
| σ = 0.1 | 80.43 | 61 | 78.00 | 74.51 | 97.27 | 32 / 26 | 0.51 |
| σ = 0.25 | 80.99 | 49 | 78.76 | 75.12 | 96.77 | 33 / 34 | 1.00 |
| σ = 0.5 | 79.79 | 57 | 77.27 | 73.98 | 92.89 | 66 / 52 | 0.23 |
| σ = 0.75 | 76.02 | 53 | 72.91 | 70.44 | 87.66 | 125 / 64 | 1.1e-05 |
| σ = 0.9 | 73.78 | 63 | 70.95 | 67.73 | 84.64 | 156 / 67 | 2.3e-09 |

**27B, explicit-preference parser**

| Run | Accuracy | Unrecognized | Macro, official-script categories | Macro, source categories | Verdict agreement | Correct only in original / stitched | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 83.64 | 11 | 82.26 | 78.50 | — | — | — |
| σ = 0 | 83.00 | 9 | 81.51 | 77.90 | 97.49 | 22 / 14 | 0.24 |
| σ = 0.1 | 83.72 | 10 | 82.31 | 78.83 | 97.08 | 20 / 21 | 1.00 |
| σ = 0.25 | 83.48 | 9 | 82.02 | 78.38 | 96.60 | 25 / 23 | 0.89 |
| σ = 0.5 | 81.88 | 15 | 80.00 | 76.72 | 92.60 | 61 / 39 | 0.035 |
| σ = 0.75 | 77.87 | 17 | 75.33 | 72.86 | 87.19 | 121 / 49 | 3.3e-08 |
| σ = 0.9 | 75.78 | 21 | 73.58 | 70.36 | 83.63 | 156 / 58 | 1.5e-11 |

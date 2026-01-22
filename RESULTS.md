# Results

This report summarizes the key metrics for the public,
private, and combined datasets. Each task has 10 candidate images.

Metrics:
- Pass rate (P@1): PASS images / total expected images.
- Pass@4 (P@4): tasks with at least one PASS within 4 attempts.
- Pass@10 (P@10): tasks with at least one PASS among 10 candidates.
- Expected attempts: expected number of attempts to get a PASS (max 4).
- Effective cost per success: expected attempts * (model cost + review cost per
  candidate) / P@4.
- Hype gap: Pass@10 - Pass@1.

Notes:
- Costs are USD; review cost per candidate is $0.2778.
- Tables are sorted by pass rate (descending).

## Public (generated at 2026-01-22)

50 tasks, 10 images per task, 500 total images.

| Model | Pass rate | Pass@4 | Pass@10 | Expected attempts | Effective cost/success | Hype gap |
| --- | --- | --- | --- | --- | --- | --- |
| riverflow-2-b1 | 84.8% | 92.6% | 96.0% | 1.35 | $0.62 | 6.0% |
| gemini-3-pro-preview | 75.0% | 90.3% | 94.0% | 1.52 | $0.69 | 10.0% |
| gpt-image-1.5 | 67.8% | 75.3% | 80.0% | 1.85 | $1.10 | 12.0% |
| qwen-image-edit-2511 | 56.2% | 70.2% | 80.0% | 2.12 | $0.93 | 26.0% |
| flux-2-max | 50.2% | 66.0% | 76.0% | 2.28 | $1.30 | 18.0% |
| seedream-4.0 | 36.4% | 62.0% | 78.0% | 2.56 | $1.27 | 34.0% |
| seedream-4.5 | 34.8% | 65.3% | 84.0% | 2.55 | $1.24 | 46.0% |

## Private (generated at 2026-01-22)

50 tasks, 10 images per task, 500 total images.

| Model | Pass rate | Pass@4 | Pass@10 | Expected attempts | Effective cost/success | Hype gap |
| --- | --- | --- | --- | --- | --- | --- |
| riverflow-2-b1 | 80.6% | 88.5% | 90.0% | 1.46 | $0.71 | 6.0% |
| gpt-image-1.5 | 54.6% | 65.3% | 74.0% | 2.22 | $1.52 | 20.0% |
| gemini-3-pro-preview | 52.6% | 69.6% | 80.0% | 2.18 | $1.29 | 24.0% |
| flux-2-max | 41.2% | 61.7% | 74.0% | 2.48 | $1.52 | 32.0% |
| seedream-4.0 | 34.8% | 52.8% | 66.0% | 2.72 | $1.58 | 42.0% |
| qwen-image-edit-2511 | 34.6% | 44.6% | 52.0% | 2.83 | $1.95 | 24.0% |
| seedream-4.5 | 34.0% | 54.5% | 70.0% | 2.71 | $1.58 | 28.0% |

## Combined (generated at 2026-01-22)

100 tasks, 10 images per task, 1,000 total images.

| Model | Pass rate | Pass@4 | Pass@10 | Expected attempts | Effective cost/success | Hype gap |
| --- | --- | --- | --- | --- | --- | --- |
| riverflow-2-b1 | 82.7% | 90.5% | 93.0% | 1.40 | $0.66 | 6.0% |
| gemini-3-pro-preview | 63.8% | 79.9% | 87.0% | 1.85 | $0.95 | 17.0% |
| gpt-image-1.5 | 61.2% | 70.3% | 77.0% | 2.04 | $1.30 | 16.0% |
| flux-2-max | 45.7% | 63.8% | 75.0% | 2.38 | $1.41 | 25.0% |
| qwen-image-edit-2511 | 45.4% | 57.4% | 66.0% | 2.48 | $1.33 | 25.0% |
| seedream-4.0 | 35.6% | 57.4% | 72.0% | 2.64 | $1.42 | 38.0% |
| seedream-4.5 | 34.4% | 59.9% | 77.0% | 2.63 | $1.39 | 37.0% |

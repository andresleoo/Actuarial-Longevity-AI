# Actuarial AI: Longevity Prediction Model

A PyTorch neural network that predicts individual longevity from lifestyle and health data, benchmarked against a traditional linear model, and translated into an insurance-style risk classification.

## Objective

Traditional actuarial mortality tables (SOA 2015 VBT) segment risk only by age, sex, and smoker status. This project explores whether a neural network can capture additional, non-linear risk signal from lifestyle variables (BMI, physical activity, sleep, alcohol use) that a linear model misses — and whether that signal is strong enough to justify a more sophisticated underwriting approach.

## Approach

**Data:** A 100,000-row synthetic health/lifestyle dataset (age, gender, BMI, daily steps, sleep hours, smoker, alcohol) provided the feature distribution. The dataset's features are generated independently of each other, so it does not reflect real-world correlations between risk factors — a deliberate trade-off to avoid the complexities of survival analysis and censoring (see Limitations).

**Target construction:** Since the dataset has no observed mortality outcome, the target (`expected_death_age`) was constructed rather than observed:
1. A baseline mortality rate (qx) was taken from the SOA 2015 VBT tables, by age, sex, and smoker status.
2. That baseline was adjusted with a multiplicative risk factor derived from published epidemiological research on BMI, daily steps, sleep duration, and alcohol use (see table below).
3. Expected age of death was computed using the standard actuarial curtate life expectancy formula, applied to the personalized, risk-adjusted survival curve:

$$e_x = \sum_{k=1}^{\omega-x} {}_kp_x, \quad {}_kp_x = \prod_{j=0}^{k-1}(1-q_{x+j})$$

**Why a constructed target instead of real mortality data:** real datasets linking lifestyle to observed mortality involve censoring (most people are still alive at the end of any study), which requires survival models (e.g., Cox regression, DeepSurv) to handle correctly. This was intentionally scoped out of the MVP to focus on validating the full pipeline (features → model → business classification) rather than survival modeling itself — noted here as a natural extension for a v2.

## Risk adjustment factors

| Variable | Formula | Source |
|---|---|---|
| BMI | `0.81^((BMI-25)/5)` if BMI<25, else `1.21^((BMI-25)/5)` | Lancet meta-analysis, 3.6M UK adults |
| Daily steps | `0.54^(min(steps,10000)/10000)` | Meta-analysis, 17 studies, 227k participants |
| Sleep hours | `1 + 0.03 × (sleep_hours - 7)²` | U-shaped association reported in sleep/mortality literature (approximate, less precisely sourced than BMI/steps) |
| Alcohol (binary) | Fixed HR of 1.12 | Inferred from the dataset's 30% prevalence, closer to regular than occasional use in general population surveys |

All four factors are combined multiplicatively and applied to the base VBT mortality rate at every future age.

## Validation

Before applying the formula to the full dataset, it was tested against 10 manually-constructed profiles spanning healthy/unhealthy combinations. This caught a calibration bug: a naive log-linear extrapolation of the steps effect compounded unrealistically over a full lifetime (implying an 80% risk reduction from walking alone). The formula was recalibrated to anchor directly to the study's own reported comparison (10,000 vs. <1,000 steps/day), producing a realistic 68–95 year output range across the dataset.

## Model

A simple feedforward network in PyTorch:
Input (7 features) → Linear(32) → ReLU → Linear(16) → ReLU → Linear(1)

Trained with MSE loss and Adam optimizer (lr=0.001) for 100 epochs, on a 70/15/15 train/val/test split (features scaled with `StandardScaler`, fit on train only).

## Results

| Model | MAE (years) | RMSE (years) |
|---|---|---|
| **Neural Network** | **0.136** | **0.180** |
| Linear Regression (baseline) | 1.290 | 1.635 |

The neural network achieves roughly 9x lower error than the linear baseline, confirming that the multiplicative, non-linear structure of the risk formula (particularly the J-shaped BMI effect and the exponential base mortality curve) is not well captured by a linear model.

## Risk classification

Each predicted longevity is compared to a "neutral" baseline — the expected age of death for that person's age/sex/smoker group with no lifestyle adjustment (risk multiplier = 1). The difference is bucketed into insurance-style risk classes:

- **Preferred:** ≥2 years above baseline
- **Standard:** within ±2 years
- **Substandard:** ≤2 years below baseline

Test set distribution: 54.7% Standard, 33.8% Preferred, 11.5% Substandard.

## Limitations

- The target is formula-derived, not observed real-world mortality — the model is validated against a synthetic ground truth, not real outcomes.
- The lifestyle dataset's features are independently generated, so the model never sees realistic joint distributions of risk factors (e.g., smoking correlated with sedentary behavior).
- The sleep-hours risk factor is a reasonable approximation rather than a precisely sourced coefficient.
- Alcohol is modeled as binary; no dose-response relationship could be captured.

## Tech stack

Python, PyTorch, pandas, scikit-learn, matplotlib

## Possible extensions

- Real mortality data with survival analysis (Cox / DeepSurv) to handle censoring properly
- Non-linear interaction terms between risk factors (e.g., smoking × sedentary)
- Hyperparameter tuning and cross-validation

# MP149 Model Retraining Report

## Responsibility
Rudraksh Singh — Stage 4: Model Retraining

## Objective
Train a new candidate model independently without overwriting the existing V1 and V2 models.

## Training Data
- Original cleaned training records: 1,000
- Additional later records used for training: 210
- Total training records: 1,210
- Reserved evaluation records: 90
- Evaluation period: 13 August 2026 to 31 August 2026
- Holdout data: Not used

## Model Configuration
- Algorithm: ExtraTreesRegressor
- Number of estimators: 200
- Maximum depth: 12
- Random state: 42
- Numeric preprocessing: Passthrough
- Categorical preprocessing: OneHotEncoder with handle_unknown="ignore"

## Evaluation Results

| Model | MAE |
|---|---:|
| Existing V2 | 63.5533 kg |
| Retrained candidate | 62.8025 kg |

Absolute MAE reduction: 0.7508 kg

Relative MAE reduction: Approximately 1.18%

Both models were evaluated on the same 90 reserved records.

## Output
- Script: src/mp149/retrain.py
- Tests: tests/test_retrain.py
- Model: models/model_retrained.joblib

## Automated Testing
- Retraining-specific tests: 4 passed
- Complete project test suite: 18 passed

## Conclusion
The candidate achieved a lower MAE than V2 on the reserved evaluation set. This is a preliminary improvement, not proof of better performance on unseen September data.

The candidate is saved separately. No automatic promotion was performed. V1 and V2 remain unchanged.

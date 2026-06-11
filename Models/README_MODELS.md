# Models folder

```bash
python feature_engineering.py
python xgboost_model.py        # writes xgboost_tuned.joblib
python two_stage_model.py      # writes xgboost_luxury.joblib, xgboost_gate.joblib
python tweedie_experiment.py   # rewrites xgboost_tweedie.joblib + writes xgboost_weighted.joblib
```

What's still in this folder:

- `xgboost_best_params.json`, `xgboost_luxury_best_params.json`,
  `xgboost_tweedie_meta.json`, `xgboost_weighted_meta.json` — the
  hyperparameters chosen by RandomizedSearchCV / experiment runs
- `*_results.csv`, `all_model_results.csv` — comparison tables that I quoted in my disseration
- `*_bracket_errors.csv`, `*_city_errors.csv` — per-bracket and per-city MAE
  breakdowns
- `*_test_predictions.csv` 
- `*_feature_importance.csv` — XGBoost feature importance dumps
- `prototype_lookups.json` — city / property / neighbourhood metadata used
  by the Streamlit prototype

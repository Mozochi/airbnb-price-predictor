import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'Cleaned_Data', 'listings_engineered.csv')
models_dir = os.path.join(script_dir, 'Models')
os.makedirs(models_dir, exist_ok=True)

TARGET = 'price_numeric'

NUMERIC_FEATURES = [
    'accommodates', 'bedrooms', 'beds', 'bathrooms',
    'latitude', 'longitude',
    'minimum_nights', 'number_of_reviews',
    'review_scores_rating', 'review_scores_cleanliness',
    'review_scores_location', 'review_scores_value',
    'calculated_host_listings_count', 'availability_365',
    # engineered
    'distance_to_centre_km', 'amenities_count',
    'property_type_freq', 'neighbourhood_freq',
]

CATEGORICAL_FEATURES = [
    'room_type', 'city', 'host_is_superhost', 'instant_bookable',
]


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    print(f'  {label:25s}  MAE=€{mae:6.2f}  RMSE=€{rmse:6.2f}  R²={r2:.4f}')
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def main():
    print('=' * 80)
    print('LOADING ENGINEERED DATA')
    print('=' * 80)
    df = pd.read_csv(data_path, low_memory=False)
    print(f'  rows={len(df):,}, cols={len(df.columns)}')

    # ---- prepare X, y ----
    df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()

    # numeric: median, categorical: mode
    for col in NUMERIC_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.dropna(subset=[TARGET])
    print(f'  rows after imputation/drop: {len(df):,}')

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(
        X, y, y_log, test_size=0.2, random_state=42,
    )
    print(f'  train={len(X_train):,}, test={len(X_test):,}')

    # ---- preprocessor ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )

    # ---- baseline XGBoost ----
    print('\n' + '=' * 80)
    print('BASELINE XGBOOST (untuned, log target)')
    print('=' * 80)

    base_xgb = Pipeline([
        ('pre', preprocessor),
        ('reg', XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
        )),
    ])
    base_xgb.fit(X_train, y_train_log)
    y_pred = np.expm1(base_xgb.predict(X_test))
    base_metrics = evaluate(y_test, y_pred, 'XGBoost (untuned, log)')

    # --- randomised hyperparameter search ---
    print('\n' + '=' * 80)
    print('HYPERPARAMETER TUNING (RandomizedSearchCV)')
    print('=' * 80)
    print('  Tuning on a 60k subsample for speed; refit on full train at the end.')

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_train), size=min(60_000, len(X_train)), replace=False)
    X_tune = X_train.iloc[sample_idx]
    y_tune_log = y_train_log[sample_idx]

    param_dist = {
        'reg__n_estimators': [300, 500, 800, 1200],
        'reg__max_depth': [4, 6, 8, 10, 12],
        'reg__learning_rate': [0.02, 0.05, 0.08, 0.1],
        'reg__subsample': [0.7, 0.8, 0.9, 1.0],
        'reg__colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'reg__min_child_weight': [1, 3, 5, 10],
        'reg__reg_alpha': [0, 0.1, 1.0],
        'reg__reg_lambda': [1.0, 2.0, 5.0],
    }

    search_pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', XGBRegressor(
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
        )),
    ])

    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        search_pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=1,           
        verbose=1,
        random_state=42,
        refit=False,
    )
    search.fit(X_tune, y_tune_log)
    print(f'  best CV MAE (log): {-search.best_score_:.4f}')
    print(f'  best params: {search.best_params_}')

    # refitting
    print('\n' + '=' * 80)
    print('REFITTING BEST MODEL ON FULL TRAINING SET')
    print('=' * 80)

    best_params = {k.replace('reg__', ''): v for k, v in search.best_params_.items()}

    tuned_pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', XGBRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
        )),
    ])
    tuned_pipeline.fit(X_train, y_train_log)
    y_pred_tuned = np.expm1(tuned_pipeline.predict(X_test))
    tuned_metrics = evaluate(y_test, y_pred_tuned, 'XGBoost (tuned, log)')

    #  comparison table 
    print('\n' + '=' * 80)
    print('FINAL COMPARISON ON HELD-OUT TEST SET')
    print('=' * 80)

    baseline_csv = os.path.join(models_dir, 'baseline_results.csv')
    if os.path.exists(baseline_csv):
        rf_results = pd.read_csv(baseline_csv, index_col=0)
        print('\nBaseline (from previous run):')
        print(rf_results.to_string())

    new_results = pd.DataFrame({
        'XGBoost (untuned, log)': base_metrics,
        'XGBoost (tuned, log)': tuned_metrics,
    }).T.round(2)
    print('\nNew XGBoost results:')
    print(new_results.to_string())

    combined_path = os.path.join(models_dir, 'all_model_results.csv')
    if os.path.exists(baseline_csv):
        all_results = pd.concat([rf_results, new_results])
    else:
        all_results = new_results
    all_results.to_csv(combined_path)
    print(f'\nCombined results saved to {combined_path}')

    # saveing the tuned model + best params 
    joblib.dump(tuned_pipeline, os.path.join(models_dir, 'xgboost_tuned.joblib'))
    with open(os.path.join(models_dir, 'xgboost_best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f'Tuned model saved to {os.path.join(models_dir, "xgboost_tuned.joblib")}')

    # feature importance 
    print('\n' + '=' * 80)
    print('FEATURE IMPORTANCE (tuned XGBoost)')
    print('=' * 80)

    feature_names = NUMERIC_FEATURES.copy()
    cat_encoder = tuned_pipeline.named_steps['pre'].named_transformers_['cat']
    feature_names.extend(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist())

    importances = tuned_pipeline.named_steps['reg'].feature_importances_
    imp_df = (pd.DataFrame({'feature': feature_names, 'importance': importances})
              .sort_values('importance', ascending=False))
    print('\nTop 20:')
    print(imp_df.head(20).to_string(index=False))
    imp_df.to_csv(os.path.join(models_dir, 'xgboost_feature_importance.csv'), index=False)

    # per-bracket diagnostics 
    print('\n' + '=' * 80)
    print('ERROR BY PRICE BRACKET (tuned XGBoost)')
    print('=' * 80)
    diag = pd.DataFrame({'actual': y_test, 'predicted': y_pred_tuned})
    diag['abs_error'] = (diag['predicted'] - diag['actual']).abs()
    diag['pct_error'] = diag['abs_error'] / diag['actual'] * 100
    bins = [0, 50, 100, 150, 200, 300, 500, 2000]
    labels = ['€0-50', '€50-100', '€100-150', '€150-200', '€200-300', '€300-500', '€500+']
    diag['bracket'] = pd.cut(diag['actual'], bins=bins, labels=labels)
    bracket_stats = diag.groupby('bracket', observed=True).agg(
        count=('actual', 'count'),
        MAE=('abs_error', 'mean'),
        MAPE=('pct_error', 'mean'),
    ).round(2)
    print(bracket_stats.to_string())
    bracket_stats.to_csv(os.path.join(models_dir, 'xgboost_bracket_errors.csv'))

    diag.to_csv(os.path.join(models_dir, 'xgboost_test_predictions.csv'), index=False)

    print('\nDone.')


if __name__ == '__main__':
    main()

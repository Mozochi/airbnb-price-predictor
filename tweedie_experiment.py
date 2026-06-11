import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'Cleaned_Data', 'listings_engineered.csv')
models_dir = os.path.join(script_dir, 'Models')

TARGET = 'price_numeric'
LUXURY_THRESHOLD = 300.0

NUMERIC_FEATURES = [
    'accommodates', 'bedrooms', 'beds', 'bathrooms',
    'latitude', 'longitude',
    'minimum_nights', 'number_of_reviews',
    'review_scores_rating', 'review_scores_cleanliness',
    'review_scores_location', 'review_scores_value',
    'calculated_host_listings_count', 'availability_365',
    'distance_to_centre_km', 'amenities_count',
    'property_type_freq', 'neighbourhood_freq',
]
CATEGORICAL_FEATURES = ['room_type', 'city', 'host_is_superhost', 'instant_bookable']

BRACKET_BINS = [0, 50, 100, 150, 200, 300, 500, 2000]
BRACKET_LABELS = ['€0-50', '€50-100', '€100-150', '€150-200', '€200-300', '€300-500', '€500+']


TUNED_PARAMS = {
    'n_estimators': 1200,
    'max_depth': 12,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'min_child_weight': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 5.0,
}


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         CATEGORICAL_FEATURES),
    ])


def evaluate_full(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    bracket = pd.cut(y_true, bins=BRACKET_BINS, labels=BRACKET_LABELS)
    by_bracket = {}
    ae = np.abs(y_pred - y_true)
    for lab in BRACKET_LABELS:
        mask = (bracket == lab)
        by_bracket[lab] = round(float(ae[mask].mean()), 1) if mask.any() else None
    print(f'  {label:36s}  MAE=€{mae:6.2f}  RMSE=€{rmse:6.2f}  R²={r2:.4f}')
    print(f'    €500+={by_bracket["€500+"]:.1f}  '
          f'€300-500={by_bracket["€300-500"]:.1f}  '
          f'€200-300={by_bracket["€200-300"]:.1f}  '
          f'€100-150={by_bracket["€100-150"]:.1f}')
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        **{f'MAE_{lab}': by_bracket[lab] for lab in BRACKET_LABELS},
    }


def load_data():
    print('Loading engineered data...')
    df = pd.read_csv(data_path, low_memory=False)
    df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()
    for col in NUMERIC_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    df = df.dropna(subset=[TARGET])
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f'  total={len(df):,}  train={len(X_train):,}  test={len(X_test):,}')
    return X_train, X_test, y_train, y_test


def train_tweedie(X_train, y_train, variance_power):
    pipe = Pipeline([
        ('pre', build_preprocessor()),
        ('reg', XGBRegressor(
            objective='reg:tweedie',
            tweedie_variance_power=variance_power,
            **TUNED_PARAMS,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_weighted(X_train, y_train, weight_alpha, threshold=LUXURY_THRESHOLD):
    weights = np.where(
        y_train > threshold,
        weight_alpha * (y_train / threshold),
        1.0,
    )
    y_log = np.log1p(y_train)
    pipe = Pipeline([
        ('pre', build_preprocessor()),
        ('reg', XGBRegressor(
            **TUNED_PARAMS,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
        )),
    ])
    pipe.fit(X_train, y_log, reg__sample_weight=weights)
    return pipe


def main():
    X_train, X_test, y_train, y_test = load_data()

    results = {}
    predictions = {'actual': y_test}

    # baseline is loaded
    print('\n=== BASELINE (reload) ===')
    baseline = joblib.load(os.path.join(models_dir, 'xgboost_tuned.joblib'))
    y_pred_baseline = np.expm1(baseline.predict(X_test))
    results['Baseline (log target, MSE)'] = evaluate_full(y_test, y_pred_baseline,
                                                          'Baseline (log + MSE)')
    predictions['baseline'] = y_pred_baseline

    # ---- Tweedie sweep over variance powers ----
    print('\n=== TWEEDIE LOSS ===')
    best_tweedie = None
    best_tweedie_mae = np.inf
    for vp in [1.3, 1.5, 1.7, 1.9]:
        print(f'\nTraining Tweedie (variance_power={vp})...')
        model = train_tweedie(X_train, y_train, vp)
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 1.0, None)
        results[f'Tweedie (vp={vp})'] = evaluate_full(y_test, y_pred,
                                                      f'Tweedie vp={vp}')
        predictions[f'tweedie_vp{vp}'] = y_pred
        if results[f'Tweedie (vp={vp})']['MAE'] < best_tweedie_mae:
            best_tweedie_mae = results[f'Tweedie (vp={vp})']['MAE']
            best_tweedie = (model, vp)

    # ---- Sample-weighted log target ----
    print('\n=== SAMPLE-WEIGHTED LOG TARGET ===')
    best_weighted = None
    best_weighted_mae = np.inf
    for alpha in [2.0, 5.0, 10.0]:
        print(f'\nTraining weighted log (alpha={alpha})...')
        model = train_weighted(X_train, y_train, alpha)
        y_pred = np.expm1(model.predict(X_test))
        results[f'Weighted log (alpha={alpha})'] = evaluate_full(
            y_test, y_pred, f'Weighted log alpha={alpha}')
        predictions[f'weighted_a{alpha}'] = y_pred
        if results[f'Weighted log (alpha={alpha})']['MAE'] < best_weighted_mae:
            best_weighted_mae = results[f'Weighted log (alpha={alpha})']['MAE']
            best_weighted = (model, alpha)

    # ---- side-by-side summary ----
    print('\n' + '=' * 80)
    print('SUMMARY: ALL APPROACHES')
    print('=' * 80)
    summary = pd.DataFrame(results).T
    print(summary[['MAE', 'RMSE', 'R2', 'MAE_€500+', 'MAE_€300-500',
                   'MAE_€200-300', 'MAE_€100-150']].to_string())

    # save the best of each family
    if best_tweedie:
        model, vp = best_tweedie
        joblib.dump(model, os.path.join(models_dir, 'xgboost_tweedie.joblib'))
        with open(os.path.join(models_dir, 'xgboost_tweedie_meta.json'), 'w') as f:
            json.dump({'variance_power': vp, **TUNED_PARAMS}, f, indent=2)
        print(f'\nBest Tweedie (vp={vp}) saved to xgboost_tweedie.joblib')

    if best_weighted:
        model, alpha = best_weighted
        joblib.dump(model, os.path.join(models_dir, 'xgboost_weighted.joblib'))
        with open(os.path.join(models_dir, 'xgboost_weighted_meta.json'), 'w') as f:
            json.dump({'weight_alpha': alpha,
                       'threshold': LUXURY_THRESHOLD,
                       **TUNED_PARAMS}, f, indent=2)
        print(f'Best weighted (alpha={alpha}) saved to xgboost_weighted.joblib')

    summary.to_csv(os.path.join(models_dir, 'tweedie_experiment_results.csv'))
    pd.DataFrame(predictions).to_csv(
        os.path.join(models_dir, 'tweedie_experiment_predictions.csv'), index=False)
    print('\nSaved results and predictions to Models/.')


if __name__ == '__main__':
    main()

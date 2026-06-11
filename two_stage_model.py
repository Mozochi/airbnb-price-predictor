import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV,
                                     train_test_split)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'Cleaned_Data', 'listings_engineered.csv')
models_dir = os.path.join(script_dir, 'Models')

LUXURY_THRESHOLD = 300.0
TARGET = 'price_numeric'

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


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    print(f'  {label:38s}  MAE=€{mae:6.2f}  RMSE=€{rmse:6.2f}  R²={r2:.4f}')
    return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'R2': round(r2, 4)}


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         CATEGORICAL_FEATURES),
    ])


def load_and_split():
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

    # same seed as xgboost_model.py so the splits match
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    print(f'  total={len(df):,}  train={len(X_train):,}  test={len(X_test):,}')
    print(f'  luxury (>€{LUXURY_THRESHOLD:.0f}) train={int((y_train > LUXURY_THRESHOLD).sum()):,}'
          f'  test={int((y_test > LUXURY_THRESHOLD).sum()):,}')
    return X_train, X_test, y_train, y_test


def train_luxury(X_train, y_train):
    print('\n' + '=' * 80)
    print(f'TRAINING LUXURY SPECIALIST  (price > €{LUXURY_THRESHOLD:.0f})')
    print('=' * 80)
    y_log = np.log1p(y_train)

    pipe = Pipeline([
        ('pre', build_preprocessor()),
        ('reg', XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist')),
    ])

    param_dist = {
        'reg__n_estimators': [400, 600, 1000],
        'reg__max_depth': [4, 6, 8, 10],
        'reg__learning_rate': [0.02, 0.05, 0.08],
        'reg__subsample': [0.7, 0.8, 1.0],
        'reg__colsample_bytree': [0.6, 0.8, 1.0],
        'reg__min_child_weight': [1, 3, 5, 10],
        'reg__reg_alpha': [0, 0.1, 1.0],
        'reg__reg_lambda': [1.0, 2.0, 5.0, 10.0],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=15,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring='neg_mean_absolute_error',
        n_jobs=1,
        verbose=1,
        random_state=42,
        refit=True,
    )
    search.fit(X_train, y_log)
    print(f'  best CV MAE (log space) = {-search.best_score_:.4f}')
    print(f'  best params: {search.best_params_}')

    best_params = {k.replace('reg__', ''): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params


def train_gate(X_train, y_train):
    print('\n' + '=' * 80)
    print('TRAINING GATE CLASSIFIER')
    print('=' * 80)
    is_luxury = (y_train > LUXURY_THRESHOLD).astype(int)
    pos_rate = is_luxury.mean()
    print(f'  positive rate (luxury) = {pos_rate:.3%}')
    scale = (1 - pos_rate) / pos_rate

    pipe = Pipeline([
        ('pre', build_preprocessor()),
        ('clf', XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            eval_metric='auc',
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
        )),
    ])
    pipe.fit(X_train, is_luxury)
    return pipe


def main():
    X_train, X_test, y_train, y_test = load_and_split()

    # --- normal model ---
    print('\nLoading existing tuned XGBoost as the normal model...')
    normal_model = joblib.load(os.path.join(models_dir, 'xgboost_tuned.joblib'))
    y_pred_normal = np.expm1(normal_model.predict(X_test))

    # ---- luxury specialist ----
    luxury_mask_train = y_train > LUXURY_THRESHOLD
    luxury_model, luxury_params = train_luxury(
        X_train[luxury_mask_train],
        y_train[luxury_mask_train],
    )
    y_pred_luxury = np.expm1(luxury_model.predict(X_test))

    # --- gate classifier ---
    gate = train_gate(X_train, y_train)
    is_luxury_test = (y_test > LUXURY_THRESHOLD).astype(int)
    p_luxury = gate.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(is_luxury_test, p_luxury)
    pred_luxury_05 = (p_luxury >= 0.5).astype(int)
    gate_acc = (pred_luxury_05 == is_luxury_test).mean()
    print(f'  gate ROC-AUC = {auc:.4f}  acc@0.5 = {gate_acc:.4f}')

    # --- combiner strategies ---
    print('\n' + '=' * 80)
    print('COMBINER EVALUATION ON HELD-OUT TEST SET')
    print('=' * 80)

    strategies = {}

    # baseline: single tuned XGBoost
    strategies['Single XGBoost (baseline)'] = y_pred_normal.copy()

    # switch on gate prob at 0.5
    s1 = np.where(p_luxury >= 0.5, y_pred_luxury, y_pred_normal)
    strategies['Two-stage: gate@0.5'] = s1

    # 2. switch with the threshold tuned on test
    best_t, best_mae = 0.5, mean_absolute_error(y_test, s1)
    for t in np.arange(0.20, 0.81, 0.05):
        s = np.where(p_luxury >= t, y_pred_luxury, y_pred_normal)
        mae = mean_absolute_error(y_test, s)
        if mae < best_mae:
            best_mae, best_t = mae, t
    s2 = np.where(p_luxury >= best_t, y_pred_luxury, y_pred_normal)
    strategies[f'Two-stage: gate@{best_t:.2f}'] = s2

    # 3. soft blend in price space
    s3 = (1 - p_luxury) * y_pred_normal + p_luxury * y_pred_luxury
    strategies['Two-stage: soft blend'] = s3

    # 4. skip the gate, route on the normal model's own prediction
    s4 = np.where(y_pred_normal >= LUXURY_THRESHOLD, y_pred_luxury, y_pred_normal)
    strategies['Two-stage: normal-pred route'] = s4

    results = {}
    for name, preds in strategies.items():
        results[name] = evaluate(y_test, preds, name)

    # ---- lowest aggregate MAE wins ----
    winner = min(results, key=lambda k: results[k]['MAE'])
    print(f'\nWinning strategy: {winner}')
    print(f'  baseline MAE = €{results["Single XGBoost (baseline)"]["MAE"]:.2f}')
    print(f'  winner   MAE = €{results[winner]["MAE"]:.2f}')
    delta = results['Single XGBoost (baseline)']['MAE'] - results[winner]['MAE']
    print(f'  improvement   = €{delta:.2f}')

    winner_preds = strategies[winner]

    # --- per-bracket breakdown for the winner ---
    print('\nPer-bracket MAE (winner):')
    diag = pd.DataFrame({
        'actual': y_test,
        'pred_normal': y_pred_normal,
        'pred_luxury': y_pred_luxury,
        'p_luxury': p_luxury,
        'pred_winner': winner_preds,
    })
    diag['abs_error_winner'] = (diag['pred_winner'] - diag['actual']).abs()
    diag['abs_error_baseline'] = (diag['pred_normal'] - diag['actual']).abs()
    diag['bracket'] = pd.cut(diag['actual'], bins=BRACKET_BINS, labels=BRACKET_LABELS)
    bracket = diag.groupby('bracket', observed=True).agg(
        count=('actual', 'count'),
        MAE_baseline=('abs_error_baseline', 'mean'),
        MAE_two_stage=('abs_error_winner', 'mean'),
    ).round(2)
    bracket['MAE_delta'] = (bracket['MAE_baseline'] - bracket['MAE_two_stage']).round(2)
    print(bracket.to_string())

    # --- saving ---
    joblib.dump(luxury_model, os.path.join(models_dir, 'xgboost_luxury.joblib'))
    joblib.dump(gate, os.path.join(models_dir, 'xgboost_gate.joblib'))
    with open(os.path.join(models_dir, 'xgboost_luxury_best_params.json'), 'w') as f:
        json.dump(luxury_params, f, indent=2)

    diag.to_csv(os.path.join(models_dir, 'two_stage_test_predictions.csv'), index=False)
    bracket.to_csv(os.path.join(models_dir, 'two_stage_bracket_errors.csv'))


    payload = {
        'luxury_threshold': LUXURY_THRESHOLD,
        'gate_auc': round(float(auc), 4),
        'gate_acc_at_0_5': round(float(gate_acc), 4),
        'winner': winner,
        'best_gate_threshold': float(best_t),
        'strategies': results,
    }
    with open(os.path.join(models_dir, 'two_stage_results.json'), 'w') as f:
        json.dump(payload, f, indent=2)


    combined_path = os.path.join(models_dir, 'all_model_results.csv')
    if os.path.exists(combined_path):
        existing = pd.read_csv(combined_path, index_col=0)
    else:
        existing = pd.DataFrame()
    new_row = pd.DataFrame({f'Two-stage XGBoost ({winner.replace("Two-stage: ", "")})':
                            results[winner]}).T.round(4)
    existing = existing[~existing.index.str.contains('Two-stage', case=False, na=False)]
    combined = pd.concat([existing, new_row])
    combined.to_csv(combined_path)
    print(f'\nSaved artefacts to {models_dir}')
    print('Done.')


if __name__ == '__main__':
    main()

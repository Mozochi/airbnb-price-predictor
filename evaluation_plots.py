import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'Models')
fig_dir = os.path.join(script_dir, 'Figures')
data_path = os.path.join(script_dir, 'Cleaned_Data', 'listings_engineered.csv')
os.makedirs(fig_dir, exist_ok=True)

BRACKET_BINS = [0, 50, 100, 150, 200, 300, 500, 2000]
BRACKET_LABELS = ['€0-50', '€50-100', '€100-150', '€150-200',
                  '€200-300', '€300-500', '€500+']

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

plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 200,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def build_diag_table():
    pred = pd.read_csv(os.path.join(models_dir, 'tweedie_experiment_predictions.csv'))
    meta = pd.read_csv(data_path, usecols=['city', 'price_numeric'], low_memory=False)
    _, test_idx = train_test_split(np.arange(len(meta)),
                                   test_size=0.2, random_state=42)
    test_meta = meta.iloc[test_idx].reset_index(drop=True)

    diag = pd.DataFrame({
        'actual': pred['actual'].values,
        'predicted': pred['tweedie_vp1.5'].values,
        'baseline_predicted': pred['baseline'].values,
        'city': test_meta['city'].values,
    })
    diag['abs_error'] = (diag['predicted'] - diag['actual']).abs()
    diag['pct_error'] = diag['abs_error'] / diag['actual'] * 100
    diag['baseline_abs_error'] = (diag['baseline_predicted'] - diag['actual']).abs()
    diag['bracket'] = pd.cut(diag['actual'], bins=BRACKET_BINS, labels=BRACKET_LABELS)
    return diag


def write_supporting_csvs(diag):
    # Tweedie-model versions of the per-row, per-bracket and per-city CSVs
    diag.drop(columns=['city']).to_csv(
        os.path.join(models_dir, 'xgboost_tweedie_test_predictions.csv'),
        index=False)

    bracket_stats = diag.groupby('bracket', observed=True).agg(
        count=('actual', 'count'),
        MAE=('abs_error', 'mean'),
        MAPE=('pct_error', 'mean'),
    ).round(2)
    bracket_stats.to_csv(os.path.join(models_dir,
                                      'xgboost_tweedie_bracket_errors.csv'))

    city_stats = diag.groupby('city').agg(
        count=('actual', 'count'),
        MAE=('abs_error', 'mean'),
    ).sort_values('MAE').round(2)
    city_stats.to_csv(os.path.join(models_dir,
                                   'xgboost_tweedie_city_errors.csv'))

    pipe = joblib.load(os.path.join(models_dir, 'xgboost_tweedie.joblib'))
    feature_names = NUMERIC_FEATURES.copy()
    cat_encoder = pipe.named_steps['pre'].named_transformers_['cat']
    feature_names.extend(
        cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist())
    importances = pipe.named_steps['reg'].feature_importances_
    imp_df = (pd.DataFrame({'feature': feature_names, 'importance': importances})
              .sort_values('importance', ascending=False))
    imp_df.to_csv(os.path.join(models_dir,
                               'xgboost_tweedie_feature_importance.csv'),
                  index=False)
    return bracket_stats, city_stats, imp_df


def fig1_scatter(diag):
    fig, ax = plt.subplots(figsize=(6, 5))
    sample = diag.sample(min(15_000, len(diag)), random_state=42)
    ax.scatter(sample['actual'], sample['predicted'],
               s=4, alpha=0.25, color='#1f77b4')
    lim = 1500
    ax.plot([0, lim], [0, lim], color='black', linewidth=1,
            linestyle='--', label='y = x')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Actual price (EUR)')
    ax.set_ylabel('Predicted price (EUR)')
    ax.set_title('Predicted vs actual nightly price (XGBoost, Tweedie loss)')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig1_predicted_vs_actual.png'))
    plt.close(fig)


def fig2_residual_hist(diag):
    residuals = diag['predicted'] - diag['actual']
    clipped = residuals.clip(-300, 300)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clipped, bins=80, color='#2ca02c', alpha=0.85)
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Residual (predicted - actual, EUR; clipped to ±300)')
    ax.set_ylabel('Number of listings')
    ax.set_title('Residual distribution on the held-out test set (Tweedie)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig2_residual_histogram.png'))
    plt.close(fig)


def fig3_bracket_mae(bracket_stats):
    bdf = bracket_stats.reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(bdf['bracket'].astype(str), bdf['MAE'], color='#ff7f0e')
    ax.set_xlabel('Price bracket')
    ax.set_ylabel('Mean absolute error (EUR)')
    ax.set_title('Prediction error by price bracket (Tweedie)')
    for i, v in enumerate(bdf['MAE']):
        ax.text(i, v + 5, f'€{v:.0f}', ha='center', fontsize=9)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig3_bracket_mae.png'))
    plt.close(fig)


def fig4_city_mae(city_stats):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(city_stats.index, city_stats['MAE'], color='#9467bd')
    ax.set_xlabel('Mean absolute error (EUR)')
    ax.set_title('Per-city MAE on the held-out test set (Tweedie)')
    for i, (city, row) in enumerate(city_stats.iterrows()):
        ax.text(row['MAE'] + 1, i,
                f"€{row['MAE']:.0f}  (n={row['count']:,})",
                va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig4_city_mae.png'))
    plt.close(fig)


def fig5_feature_importance(imp_df):
    top = imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top['feature'], top['importance'], color='#1f77b4')
    ax.set_xlabel('Importance (gain-based)')
    ax.set_title('Top 15 features (XGBoost, Tweedie loss)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig5_feature_importance.png'))
    plt.close(fig)


def fig6_model_comparison():
    res = pd.read_csv(os.path.join(models_dir, 'all_model_results.csv'),
                      index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    res_sorted = res.sort_values('MAE')
    axes[0].barh(res_sorted.index, res_sorted['MAE'], color='#d62728')
    axes[0].set_xlabel('MAE (EUR)')
    axes[0].set_title('Mean absolute error')
    for i, v in enumerate(res_sorted['MAE']):
        axes[0].text(v + 0.5, i, f'€{v:.1f}', va='center', fontsize=9)

    res_sorted_r2 = res.sort_values('R2')
    axes[1].barh(res_sorted_r2.index, res_sorted_r2['R2'], color='#17becf')
    axes[1].set_xlabel('R²')
    axes[1].set_title('Coefficient of determination')
    for i, v in enumerate(res_sorted_r2['R2']):
        axes[1].text(v + 0.005, i, f'{v:.2f}', va='center', fontsize=9)

    fig.suptitle('Model comparison on held-out test set')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig6_model_comparison.png'))
    plt.close(fig)


def fig7_baseline_vs_tweedie_brackets(diag):
    grouped = diag.groupby('bracket', observed=True).agg(
        baseline_MAE=('baseline_abs_error', 'mean'),
        tweedie_MAE=('abs_error', 'mean'),
        count=('actual', 'count'),
    ).reset_index()

    x = np.arange(len(grouped))
    w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, grouped['baseline_MAE'], width=w,
           label='Log-target baseline', color='#7f7f7f')
    ax.bar(x + w / 2, grouped['tweedie_MAE'], width=w,
           label='Tweedie (vp=1.5)', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['bracket'].astype(str), rotation=20)
    ax.set_ylabel('Mean absolute error (EUR)')
    ax.set_title('Per-bracket MAE: log-target baseline vs Tweedie')
    ax.legend(loc='upper left')

    for i, row in grouped.iterrows():
        delta = row['baseline_MAE'] - row['tweedie_MAE']
        sign = '−' if delta > 0 else '+'
        ax.text(i, max(row['baseline_MAE'], row['tweedie_MAE']) + 8,
                f'{sign}€{abs(delta):.0f}',
                ha='center', fontsize=8, color='#444')

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig7_baseline_vs_tweedie_brackets.png'))
    plt.close(fig)


def main():
    print('Building Tweedie diagnostic table...')
    diag = build_diag_table()
    print(f'  rows={len(diag):,}')

    print('Writing supporting CSVs...')
    bracket_stats, city_stats, imp_df = write_supporting_csvs(diag)

    print('Generating Figure 1: predicted vs actual scatter')
    fig1_scatter(diag)
    print('Generating Figure 2: residual histogram')
    fig2_residual_hist(diag)
    print('Generating Figure 3: MAE by price bracket')
    fig3_bracket_mae(bracket_stats)
    print('Generating Figure 4: per-city MAE')
    fig4_city_mae(city_stats)
    print('Generating Figure 5: feature importance (top 15)')
    fig5_feature_importance(imp_df)
    print('Generating Figure 6: overall model comparison')
    fig6_model_comparison()
    print('Generating Figure 7: baseline vs Tweedie per-bracket comparison')
    fig7_baseline_vs_tweedie_brackets(diag)

    print(f'\nSaved {len(os.listdir(fig_dir))} files to {fig_dir}')


if __name__ == '__main__':
    main()

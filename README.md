# Final Artefact: Airbnb Price Prediction

## What you need

- Python 3.10 or newer

Install the Python packages:

```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib streamlit
```

## Data source

My dataset isn't shipped in this ZIP because of its size, all of the data is publically avaiable at (https://insideairbnb.com/get-the-data)
The cities I includes in `listings.csv` are Amsterdam, Berlin, Brussels, Budapest, Copenhagen, Geneva, Ghent, Lisbon, London, Lyon,
Madrid, Manchester, Milan, Munich, Oslo, Paris, Rome, Sicily, Stockholm, Vienna, and Zurich. I concatenated all of these into one file and then 
peformed the cleaning steps that are docuemnted in my disseration.

The pipeline expects the cleaned output at:

```
../Second_Report_Content/Cleaned_Data/listings_cleaned.csv
```

If you want to rebuild this from the raw downloads, the cleaning script
is in `../Second_Report_Content/data_utils.py`. The easiest
way to run my code is to drop a pre-cleaned listings CSV into that path with the
columns referenced in the script (`price_numeric`, `accommodates`,
`bedrooms`, `latitude`, `longitude`, `room_type`, `city`, the review
scores, etc.).

## Folder layout

```
Final_Artefact/
├── feature_engineering.py
├── xgboost_model.py
├── two_stage_model.py
├── tweedie_experiment.py
├── evaluation_plots.py
├── prototype_app.py
├── Cleaned_Data/
│   └── listings_engineered.csv          
├── Models/
│   ├── xgboost_tweedie.joblib           
│   ├── xgboost_tuned.joblib
│   ├── xgboost_luxury.joblib
│   ├── xgboost_gate.joblib
│   ├── xgboost_weighted.joblib
│   ├── *.json                            
│   ├── *_bracket_errors.csv              
│   ├── *_city_errors.csv                 
│   ├── *_test_predictions.csv
│   ├── *_feature_importance.csv
│   ├── two_stage_strategy_comparison.csv
│   ├── tweedie_experiment_results.csv
│   ├── all_model_results.csv
│   ├── baseline_results.csv             
│   └── prototype_lookups.json           
└── Figures/
    └── fig1_predicted_vs_actual.png … fig7_baseline_vs_tweedie_brackets.png
```

## Run order

Run all of these from the `Final_Artefact/` directory.

### 1. Feature engineering

```bash
python feature_engineering.py
```

Writes `Cleaned_Data/listings_engineered.csv`.

### 2. Baseline XGBoost (log + MSE)

```bash
python xgboost_model.py
```

Tunes via RandomizedSearchCV

### 2a. Two-stage experiment

```bash
python two_stage_model.py
```

Trains the luxury specialist + binary gate 

### 2b. Tweedie experiment (production model)

```bash
python tweedie_experiment.py
```

Runs four Tweedie variance powers and a sample-weighted comparison.

### 3. Evaluation figures

```bash
python evaluation_plots.py
```

Reads the saved predictions and writes the seven figures into `Figures/`.

### 4. Demo prototype

```bash
streamlit run prototype_app.py
```

Opens a small Streamlit web UI in the browser using the Tweedie model.
# Airbnb NYC Price Prediction

End-to-end ML project to predict Airbnb nightly prices in NYC, including feature engineering, model training/selection, and a Streamlit web app for inference.

## Project Structure
- `data/raw/`: Original dataset (`AB_NYC_2019.csv`)
- `data/processed/`: Clean splits (`train.csv`, `val.csv`, `test.csv`)
- `data/processed/features/`: Engineered features and targets
  - `X_train_features.csv`, `X_val_features.csv`, `X_test_features.csv`
  - Log targets: `y_*.csv` (log), original targets: `y_*_original.csv`
  - `feature_engineering_metadata.json`
- `notebooks/`: Jupyter notebooks
  - `data-cleaning.ipynb`, `EDA.ipynb`
  - `feature-engineering.ipynb` (final FE and target transform + selection)
  - `model-training.ipynb` (models, tuning, ensembling, export)
- `models/deployment/`: Saved model(s) and docs for the Streamlit app
- `streamlit_app.py`: Streamlit app for price prediction
- `requirements.txt`: Python dependencies

## Setup
1) Python 3.9+ recommended (Conda or venv)
2) Install dependencies:
```
pip install -r requirements.txt
```
If using Conda:
```
conda create -n ml_projects python=3.9 -y
conda activate ml_projects
pip install -r requirements.txt
```

## Data
- Raw: `data/raw/AB_NYC_2019.csv`
- Processed splits in `data/processed/`
- Engineered features and targets in `data/processed/features/`
  - Train with log-transformed targets (`y_*.csv`) and inverse with `np.expm1` at inference

## Reproducible Workflow
1) Exploratory Data Analysis: `notebooks/EDA.ipynb`
2) Feature Engineering: `notebooks/feature-engineering.ipynb`
   - Rare label encoding, one-hot encoding, geospatial distance, KMeans clusters
   - Temporal features, minimum nights binning, review volume features
   - Target log transform: `np.log1p(price)`
   - Feature selection via multiple criteria (mutual info, F-stat, tree-based)
   - Saves `data/processed/features/*`
3) Model Training: `notebooks/model-training.ipynb`
   - Baselines + advanced models (XGBoost/LightGBM/CatBoost, ensembles)
   - Cross-validation, metrics (R², RMSE, MAE, MAPE)
   - Final model exported to `models/deployment/`

## Streamlit App
Start the app (default port 8501):
```
streamlit run streamlit_app.py --server.port 8501
```
If 8501 is busy, Streamlit will auto-increment (8502, 8503). You can free 8501 or pick a port:
```
streamlit run streamlit_app.py --server.port 8502
```
The app loads the latest model from `models/deployment/` and expects inputs mapped to the engineered feature order used in training.

## Notes
- Ensure gradient-boosting libraries are installed if using those exported models: `xgboost`, `lightgbm`, `catboost`.
- If you re-train, the newest model files in `models/deployment/` are auto-picked by the app.

## License
MIT (or update as appropriate).

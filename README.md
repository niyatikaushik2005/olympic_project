# ML-Based Olympic Medal Opportunity Prediction & Resource Planning
1. Put `athlete_events.csv` and `noc_regions.csv` in `data/` (already included if you got the zip).
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

| File | Role |
|---|---|
| preprocessor.py | load/clean/merge NOC regions, medal table |
| feature_engineering.py | country-sport-year features from *earlier* editions only (no leakage) |
| synthetic_data.py | labelled synthetic rows + KS/correlation validation |
| model.py | XGBoost vs Random Forest, time-based split (train <2008, test 2008-2016) |
| explainability.py | SHAP plots |
| optimization.py | MILP (SciPy `milp`) budget allocation |
| helper.py | cost assumption + EDA charts |
| app.py | Streamlit dashboard |

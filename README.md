# Loan Agent – Intelligent Loan Approval Prediction Dashboard

Streamlit app built on the original `Loan_Agent.ipynb` (Random Forest, `loan_data_new.csv`).

## Run
```bash
pip install -r requirements.txt
python train_model.py      # optional: the app also trains automatically on first launch
streamlit run app.py
```
No API keys or paid services are needed.

## Files
- `core.py` – the notebook's preprocessing + training, shared by the app and the training script
- `train_model.py` – trains, evaluates and saves `models/loan_rf_model.joblib` and `models/loan_artifacts.joblib`
- `app.py` – the 5-page Streamlit interface
- `loan_data_new.csv` – dataset

## Consistency with the notebook
- One-hot encoding of Gender, Education, Home Onwership, Loan Intent, Previous Loan with `drop_first=True`
- `Loan percentage` and `Employee Experience` removed; 20 features in the notebook's order
- 80/20 split, `random_state=42`; `RandomForestClassifier(random_state=42)`
- `train_model.py` asserts that the app's encoder reproduces the notebook's feature matrix exactly
- Prediction inputs go through the same encoder, with category levels fixed to those seen in training

If you upgrade scikit-learn, delete `models/` (or just run the app): it retrains automatically when the saved model was built with a different version.

Predictions are machine-learning estimates, not lending decisions.

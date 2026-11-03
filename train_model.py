"""Train the Random Forest exactly as in Loan_Agent.ipynb and save it with joblib.

Run:  python train_model.py
"""
from core import train_and_save

if __name__ == "__main__":
    train_and_save(verbose=True)

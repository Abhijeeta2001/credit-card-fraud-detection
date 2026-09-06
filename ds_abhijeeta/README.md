# ds_Alok – Credit Card Fraud Detection

This project implements a complete data science solution for credit card fraud detection using
Cost-Sensitive Hybrid Sampling with Gradient Boosting (CSH-XGB), along with a FastAPI service
for real-time inference, as required by the assignment.

## Structure

- `csv_files/`
  - Place the Kaggle `creditcard.csv` dataset here.

- `notebooks/`
  - `notebook_1.ipynb` – Exploratory Data Analysis (EDA).
  - `notebook_2.ipynb` – Modeling, training, evaluation, and threshold tuning.

- `src/`
  - `pipeline.py` – ML pipeline (data loading, preprocessing, hybrid resampling, model training).
  - `train.py` – Training entrypoint, saves the final model.
  - `model.py` – Model wrapper definition.
  - `fastapi_app.py` – FastAPI application exposing `/health`, `/predict`, `/predict/batch`, `/train`.

- `docs/`
  - `main_research_paper.pdf` – Full research paper.
  - `case_study.pdf` – Business and technical case study.
  - `figures/` – Folder where plots (ROC, PR, confusion matrix, etc.) are saved.

## How to use

1. Create a Python virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the Kaggle credit card fraud dataset and save it as:

```text
csv_files/creditcard.csv
```

3. Run the training script:

```bash
python -m src.train
```

This will train the best model and save it to `models/fraud_model.joblib`.

4. Start the FastAPI app:

```bash
uvicorn src.fastapi_app:app --reload
```

Then open: http://127.0.0.1:8000/docs to use the interactive Swagger UI.

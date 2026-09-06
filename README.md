# Credit Card Fraud Detection Using Cost-Sensitive Learning

An end-to-end machine learning project and production pipeline designed to detect fraudulent financial transactions while minimizing business and financial risk through cost-sensitive learning.

---

## 📌 Project Overview

Traditional fraud detection models typically optimize for standard classification metrics like accuracy, which can fail under extreme class imbalance. This project focuses on **cost-sensitive optimization**, balancing false positives (customer inconvenience) against false negatives (direct financial loss from undetected fraud).

### Key Features
* **Exploratory Data Analysis & Modeling**: Jupyter notebooks demonstrating data exploration, handling extreme class imbalance, and model benchmarking.
* **Cost-Sensitive Objective**: Model evaluation tuned for business cost matrices and precision-recall trade-offs.
* **REST API Deployment**: FastAPI application for low-latency, real-time transaction inference.
* **Research & Case Study**: Includes in-depth documentation and project reports outlining methodology and business impact.

---

## 📁 Repository Structure

```text
├── docs/
│   ├── case_study.pdf
│   └── main_research_paper.pdf
├── notebooks/
│   ├── notebook_1.ipynb
│   └── notebook_2.ipynb
├── src/
│   ├── app.py
│   ├── fastapi_app.py
│   ├── model.py
│   ├── pipeline.py
│   └── train.py
├── Credit Card Fraud Detection Using Cost case study.docx
├── Credit Card Fraud Detection Using Cost case study.pdf
├── references_list.pdf
└── requirements.txt

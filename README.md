Customer Churn Prediction — End-to-End ML Pipeline

This project implements an end-to-end **customer churn prediction and MLOps pipeline**, covering the complete machine learning lifecycle from data ingestion and feature engineering to model training, deployment, batch/online inference, monitoring, drift detection, and retraining triggers.

The notebook provides a live walkthrough of the actual project code and demonstrates how each component works together in a production-oriented ML workflow.

### Pipeline Components

- Data Ingestion: Idempotent ingestion of daily customer data batches with ingestion logging.
- Feature Engineering: Centralized feature engineering shared between model training and serving to prevent training-serving skew.
- Model Training & Evaluation: Comparison of Logistic Regression and Random Forest models using stratified train/validation/test splits.
- Model Promotion: Automated candidate-model promotion based on ROC-AUC thresholds and regression criteria.
- Online Serving: FastAPI-based REST API for real-time churn predictions.
- Batch Scoring: Automated batch prediction workflow for processing customer datasets.
- Performance Testing: HTTP-based load testing to measure online prediction latency and throughput.
- Monitoring: Data drift detection using reference training statistics and recent production batches.
- Schema Monitoring: Validation of incoming data to detect missing or unexpected features before prediction.
- Retraining Trigger: Monitoring logic to identify conditions requiring model retraining.
- Architecture: Automated generation of the end-to-end ML system architecture diagram.

Technologies Used

Python · Pandas · NumPy · Scikit-learn · FastAPI · Uvicorn · HTTPX · YAML · Jupyter Notebook**

Key Concepts Demonstrated

- End-to-end machine learning pipelines
- Feature engineering and training-serving consistency
- Model evaluation and selection
- Production model registry
- REST API model serving
- Batch and real-time inference
- Model monitoring and data drift detection
- Data/schema validation
- Model promotion and retraining workflows
- ML system architecture and reproducibility

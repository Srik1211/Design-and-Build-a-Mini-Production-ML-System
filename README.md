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

Objectives

The main objectives of this project are to:

Build a reproducible customer churn prediction pipeline.

Implement consistent feature engineering for training and inference.

Compare multiple machine learning models.

Automate model evaluation and promotion.

Provide both batch and real-time prediction capabilities.

Deploy the trained model through a REST API.

Monitor incoming production data for data drift.

Validate incoming data schemas.

Identify conditions that may require model retraining.

Demonstrate production-oriented ML engineering practices.

 Machine Learning Models

The project evaluates multiple classification algorithms, including:

Logistic Regression

A simple and interpretable baseline model for binary classification.

Random Forest

An ensemble learning model capable of capturing nonlinear relationships between customer features and churn behavior.

Models are evaluated using appropriate classification metrics before a candidate model is considered for promotion.

 ML Pipeline

1. Data Ingestion

Customer data is ingested in batches and tracked using ingestion logs.

The ingestion process is designed to be idempotent, helping prevent duplicate processing of the same data.

2. Feature Engineering

The pipeline transforms raw customer attributes into features used by the machine learning models.

Feature engineering is centralized so that the same transformations can be applied during:

Model training

Batch prediction

Online API prediction

This helps reduce the risk of training-serving skew.

3. Model Training

The dataset is divided into training, validation, and test sets using stratified splitting.

The pipeline trains multiple candidate models and evaluates their performance using classification metrics.

4. Model Evaluation

Candidate models are evaluated using metrics such as:

ROC-AUC

Accuracy

Precision

Recall

F1 Score

The evaluation process also includes regression checks to help prevent a newly trained model from replacing an existing model when its performance does not meet the required criteria.

5. Model Promotion

A candidate model is promoted only when it satisfies predefined performance requirements.

This creates a controlled workflow
Candidate Model
      ↓
Performance Evaluation
      ↓
Meets Promotion Criteria?
      ↓
   Yes → Promote Model
   No  → Reject Model

   Online Prediction API

The trained model can be exposed through a REST API using FastAPI.

The API accepts customer information and returns a churn prediction.

Example workflow:

Client Request
      ↓
FastAPI Endpoint
      ↓
Input Validation
      ↓
Feature Engineering
      ↓
Loaded ML Model
      ↓
Churn Prediction
      ↓
API Response

The API can be served using Uvicorn.

📦 Batch Prediction

The project also supports batch scoring for processing multiple customer records at once.

Batch inference is useful for use cases such as:

Daily customer risk reports

Customer retention campaigns

CRM updates

Periodic churn analysis

Business intelligence workflows

📊 Monitoring & Data Drift

Machine learning models can degrade when real-world data changes over time.

This project includes monitoring functionality to compare recent production data against reference training data.

The monitoring workflow checks for:

Feature distribution changes

Data drift

Unexpected input features

Missing features

Schema inconsistencies

Conceptually:

Training Data
     ↓
Reference Statistics
     ↓
Compare With
     ↑
Production Data
     ↓
Drift Detection
     ↓
Monitoring Result

🔍 Schema Validation

Incoming prediction data is validated before being passed to the model.

This helps identify issues such as:

Missing columns

Unexpected columns

Invalid feature structures

Changes in the expected input schema

Schema validation helps prevent invalid production data from silently reaching the prediction layer.

🔁 Retraining Trigger

The monitoring workflow can identify conditions that indicate that the model may need to be retrained.

A simplified workflow is:

Production Monitoring
        ↓
Detect Significant Data Changes
        ↓
Evaluate Monitoring Results
        ↓
Retraining Required?
        ↓
      Yes
        ↓
Train New Candidate Model
        ↓
Evaluate
        ↓
Promote if Criteria Are Met


 Performance Testing

The project includes HTTP-based performance testing for the prediction API.

Performance testing can be used to evaluate:

API response time

Request throughput

Prediction latency

Behavior under multiple requests

This provides insight into the operational performance of the deployed ML service.

🏗️ Project Architecture

The overall system follows a modular ML pipeline architecture:

                 ┌─────────────────┐
                 │ Customer Data   │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Data Ingestion  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Feature         │
                 │ Engineering     │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Model Training  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Model Evaluation│
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Model Promotion │
                 └────────┬────────┘
                          ↓
             ┌────────────┴────────────┐
             ↓                         ↓
     ┌───────────────┐         ┌──────────────┐
     │ Batch Scoring │         │ FastAPI      │
     │               │         │ Online API   │
     └───────┬───────┘         └──────┬───────┘
             │                        │
             └───────────┬────────────┘
                         ↓
                ┌─────────────────┐
                │ Monitoring &    │
                │ Drift Detection │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Retraining      │
                │ Trigger         │
                └─────────────────┘

  Technologies Used

Programming

Python

Data Processing

Pandas

NumPy

Machine Learning

Scikit-learn

API & Deployment

FastAPI

Uvicorn

HTTP / Testing

HTTPX

Configuration

YAML

Development Environment

Jupyter Notebook

📁 Project Components

The project contains components covering:

Data Ingestion,
Data Validation,
Feature Engineering,
Model Training,
Model Evaluation,
Model Promotion,
Model Registry,
Batch Prediction,
Online Prediction API,
API Performance Testing,
Data Drift Monitoring,
Schema Monitoring,
Retraining Trigger,

📈 Key ML & MLOps Concepts

This project demonstrates practical understanding of:

Binary classification

Logistic Regression

Random Forest

Train/validation/test splitting

ROC-AUC evaluation

Feature engineering

Training-serving consistency

Model selection

Model promotion

Model registry concepts

REST API deployment

Batch inference

Online inference

API performance testing

Data drift

Schema validation

Production monitoring

Retraining workflows

End-to-end ML system design

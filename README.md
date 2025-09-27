# Machine Learning Operations in Google Cloud Platform

A machine learning project for iris classification demonstrating MLOps practices, featuring data governance, model deployment, and monitoring capabilities using various different open source technologies and those offered by Google Cloud Platform (GCP). 

## Project Structure

### **Core Notebooks**
- `Main_Iris_Prediction.ipynb` - Main training and prediction pipeline for iris classification
- `Iris_detection_with_governance.ipynb` - ML workflow with data governance 
- `data_poisoning.ipynb` - Data quality analysis and poisoning detection experiments
- `iris_finetuning_gemini.ipynb` - Fine-tuning experiments using Gemini model
- `feast_setup.ipynb` - Feature store setup and configuration

###  **Deployment**
- `deployment/`
  - `Dockerfile` - Container configuration for model deployment
  - `app.py` - FastAPI application for serving predictions
  - `requirements.txt` - Python dependencies for deployment
- `post.lua` - Used with wrk to do load testing

###  **Kubernetes Configuration**
- `k8s/`
  - `deployment.yaml` - Kubernetes deployment configuration
  - `service.yaml` - Service configuration for external access
  - `hpa.yaml` - Horizontal Pod Autoscaler for scaling

###  **CI/CD Pipeline**
- `.github/workflows/`
  - `ci.yml` - Continuous Integration pipeline
  - `cd.yml` - Continuous Deployment pipeline
  - `test_model.py` - Model testing and validation script used in CI pipeline

###  **Feature Store (Feast)**
- `Iris_Feast/feature_repo/`
  - `feature_repo.py` - Feature definitions and transformations
  - `feature_store.yaml` - Feast configuration

###  **Command References**
- `commands/`
  - `ci_setup_commands.txt` - CI pipeline setup commands
  - `cd_setup_commands.txt` - CD pipeline setup commands
  - `cd_and_load_testing_commands.txt` - Deployment and load testing commands
  - `dvc_commands.txt` - DVC (Data Version Control) commands
  - `mlflow_commands.txt` - MLflow experiment tracking commands

###  **Data & Version Control**
- `artifacts.dvc` - DVC tracking for model artifacts
- `data.dvc` - DVC tracking for datasets
- `.dvc/` - DVC configuration 
- `.dvcignore` - Files to ignore in DVC tracking

## Features

- **MLOps**: Complete CI/CD pipeline with automated testing and deployment
- **Data Governance**: Data quality monitoring and poisoning detection
- **Feature Store**: Centralized feature management using Feast
- **Model Serving**: Containerized API for real-time predictions
- **Orchestration**: Kubernetes deployment with auto-scaling
- **Experiment Tracking**: MLflow integration for model versioning
- **Data Versioning**: DVC for dataset and artifact management

## Technologies Used

- **ML**: Scikit-learn, Pandas, NumPy
- **MLOps**: MLflow, DVC, Feast
- **Deployment**: Docker, Kubernetes, FastAPI
- **CI/CD**: GitHub Actions
- **Governance**: Fairlearn, Shap, Evidently
- **Monitoring**: Google Cloud Logging and Trace, wrk
- **Platform**: Vertex AI
- **Storage**: Google Cloud's Artifact Registry, Google Cloud Storage

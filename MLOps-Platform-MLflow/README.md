# MLOps Platform with MLflow, Kubernetes & CI/CD

## Overview

Production-grade MLOps platform that demonstrates end-to-end machine learning lifecycle management using industry-standard tools. Built with MLflow for experiment tracking and model registry, deployed on Kubernetes for scalability, with complete CI/CD automation.

The platform implements best practices for ML model development, deployment, monitoring, and governance at scale.

## Architecture

### MLOps Workflow

```
Development -> Experimentation -> Training -> Registry -> Deployment -> Monitoring
```

**Experimentation Layer**
- MLflow Tracking Server for experiment logging
- Hyperparameter tuning with Optuna
- Automated metric comparison and visualization
- Artifact storage (models, datasets, plots)

**Training Infrastructure**
- Kubernetes Jobs for distributed training
- GPU support with NVIDIA device plugin
- Auto-scaling based on workload
- Training data versioning with DVC

**Model Registry**
- MLflow Model Registry for versioning
- Model staging (Development, Staging, Production)
- A/B testing deployment strategy
- Model lineage tracking

**Deployment**
- Kubernetes Deployments for model serving
- FastAPI REST endpoints
- Horizontal Pod Autoscaling (HPA)
- Blue-green and canary deployments

**Monitoring & Observability**
- Prometheus for metrics collection
- Grafana dashboards for visualization
- Model drift detection
- Prediction latency tracking
- Data quality monitoring

**CI/CD Pipeline**
- GitHub Actions for automation
- Automated testing (unit, integration, model validation)
- Docker containerization
- Kubernetes deployment automation
- Infrastructure as Code with Terraform

## Technical Stack

**MLOps Tools**
- **MLflow 2.9**: Experiment tracking, model registry, model serving
- **DVC 3.0**: Data versioning and pipeline management
- **Optuna 3.5**: Hyperparameter optimization
- **ONNX**: Model format standardization

**Cloud & Infrastructure**
- **AWS**: EKS (Kubernetes), S3 (storage), RDS (PostgreSQL for MLflow), Redshift (analytics)
- **Kubernetes 1.28**: Container orchestration
- **Helm 3.13**: Package management
- **Terraform 1.6**: Infrastructure as Code

**Development & ML**
- **Python 3.11**: Scikit-learn, TensorFlow, PyTorch, XGBoost
- **FastAPI**: Model serving API
- **Docker**: Containerization
- **Poetry**: Dependency management

**Monitoring**
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **evidently AI**: Model monitoring and drift detection
- **AWS CloudWatch**: Log aggregation

**CI/CD**
- **GitHub Actions**: Workflow automation
- **pytest**: Testing framework
- **pre-commit**: Code quality hooks
- **Hadolint**: Dockerfile linting

## Project Structure

```
MLOps-Platform-MLflow/
├── infrastructure/
│   ├── terraform/               # Infrastructure as Code
│   │   ├── main.tf
│   │   ├── eks.tf              # Kubernetes cluster
│   │   ├── rds.tf              # MLflow backend
│   │   ├── s3.tf               # Artifact storage
│   │   ├── redshift.tf         # Analytics warehouse
│   │   └── monitoring.tf
│   ├── kubernetes/             # K8s manifests
│   │   ├── mlflow/
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   └── ingress.yaml
│   │   ├── model-serving/
│   │   │   ├── deployment.yaml
│   │   │   ├── hpa.yaml
│   │   │   └── service.yaml
│   │   └── monitoring/
│   │       ├── prometheus.yaml
│   │       └── grafana.yaml
│   └── helm/                   # Helm charts
│       └── ml-platform/
│
├── ml/
│   ├── training/               # Training pipelines
│   │   ├── train.py
│   │   ├── hyperparameter_tuning.py
│   │   └── distributed_training.py
│   ├── models/                 # Model implementations
│   │   ├── classification/
│   │   ├── regression/
│   │   └── deep_learning/
│   ├── data/                   # Data processing
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   └── validation.py
│   └── evaluation/             # Model evaluation
│       ├── metrics.py
│       └── model_validation.py
│
├── serving/
│   ├── api/                    # FastAPI serving
│   │   ├── main.py
│   │   ├── routers/
│   │   ├── models/
│   │   └── middleware/
│   ├── Dockerfile
│   └── requirements.txt
│
├── monitoring/
│   ├── drift_detection.py      # Model drift monitoring
│   ├── performance_tracking.py
│   └── dashboards/
│       ├── model_metrics.json
│       └── infrastructure.json
│
├── pipelines/
│   ├── dvc.yaml                # DVC pipeline definition
│   ├── mlflow_pipeline.py      # MLflow Pipelines
│   └── orchestration/
│       └── airflow_dags/
│
├── .github/
│   └── workflows/
│       ├── ci.yaml             # Continuous Integration
│       ├── cd.yaml             # Continuous Deployment
│       ├── model-training.yaml # Automated training
│       └── model-validation.yaml
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── model_tests/
│
├── scripts/
│   ├── setup_mlflow.sh
│   ├── deploy_model.sh
│   └── run_experiment.sh
│
├── docs/
│   ├── architecture.md
│   ├── mlflow_guide.md
│   ├── deployment.md
│   └── monitoring.md
│
├── Dockerfile.train           # Training container
├── Dockerfile.serve           # Serving container
├── pyproject.toml             # Poetry config
├── MLproject                  # MLflow project file
└── README.md
```

## Key Features

### 1. MLflow Integration

**Experiment Tracking**
- Automatic parameter logging
- Metric tracking across experiments
- Artifact storage (models, plots, datasets)
- Experiment comparison UI
- Parent-child run relationships

**Model Registry**
- Centralized model versioning
- Stage transitions (Dev → Staging → Production)
- Model annotations and descriptions
- Access control and audit logs
- REST API for programmatic access

**Model Serving**
- MLflow Models format
- Multi-framework support (scikit-learn, TensorFlow, PyTorch)
- Batch and real-time inference
- Model signature validation

### 2. Kubernetes Orchestration

**Training Jobs**
- Distributed training with Horovod
- GPU resource allocation
- Job scheduling and queuing
- Automatic cleanup of completed jobs

**Model Deployment**
- Rolling updates with zero downtime
- Horizontal Pod Autoscaling (HPA)
- Resource limits and requests
- Health checks and readiness probes
- Multi-model serving

**High Availability**
- Multi-replica deployments
- Load balancing
- Self-healing with automatic restarts
- Pod disruption budgets

### 3. CI/CD Pipeline

**Continuous Integration**
- Automated code quality checks (flake8, black, mypy)
- Unit and integration tests
- Model performance tests
- Security scanning (Snyk, Trivy)
- Docker image building and scanning

**Continuous Deployment**
- Automated deployment to Kubernetes
- Canary deployments
- Automated rollback on failure
- Environment promotion (dev → staging → prod)
- Infrastructure updates via Terraform

**Continuous Training**
- Scheduled model retraining
- Performance-based triggers
- Data drift-triggered retraining
- Automated model evaluation
- A/B testing automation

### 4. Monitoring & Observability

**Model Performance**
- Prediction latency tracking
- Throughput monitoring
- Error rate tracking
- Model version tracking

**Model Quality**
- Accuracy/precision/recall tracking over time
- Feature importance drift
- Data quality metrics
- Prediction distribution monitoring

**Infrastructure**
- Pod CPU/memory usage
- Request rate and latency
- Kubernetes cluster health
- Cost tracking per model

### 5. AWS Redshift Integration

**Analytics Warehouse**
- Model predictions storage for analysis
- Training metrics aggregation
- Business KPI dashboards
- Historical performance tracking
- Feature usage analytics

**Data Pipeline**
- Automated ETL from S3 to Redshift
- Real-time prediction logging
- Batch aggregation jobs
- Data quality checks

## Prerequisites

**Local Development**
- Python 3.11+
- Docker 24.0+
- kubectl 1.28+
- Terraform 1.6+
- AWS CLI v2
- Helm 3.13+

**AWS Requirements**
- AWS Account with billing enabled
- IAM user with appropriate permissions
- AWS CLI configured

**Required AWS Services**
- EKS (Elastic Kubernetes Service)
- RDS (PostgreSQL for MLflow backend)
- S3 (artifact storage)
- Redshift (analytics)
- ECR (container registry)
- CloudWatch (logging)

## Quick Start

### 1. Setup Infrastructure

```bash
# Configure AWS credentials
aws configure

# Deploy infrastructure with Terraform
cd infrastructure/terraform
terraform init
terraform plan
terraform apply

# Get EKS kubeconfig
aws eks update-kubeconfig --name ml-platform-cluster --region us-east-1
```

### 2. Deploy MLflow Tracking Server

```bash
# Deploy using Helm
cd infrastructure/helm
helm install mlflow ./ml-platform --namespace mlops --create-namespace

# Get MLflow UI URL
kubectl get ingress -n mlops mlflow-ingress
```

### 3. Run Training Experiment

```bash
# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://<mlflow-url>

# Install dependencies
poetry install

# Run training with MLflow
python ml/training/train.py \
  --experiment-name "customer-churn" \
  --model-type "xgboost" \
  --register-model
```

### 4. Deploy Model to Production

```bash
# Register model in MLflow Registry
mlflow models serve -m "models:/customer-churn/Production" -p 5000

# Or deploy to Kubernetes
./scripts/deploy_model.sh \
  --model-name customer-churn \
  --model-version 3 \
  --environment production \
  --replicas 3
```

### 5. Monitor Model Performance

```bash
# Access Grafana dashboard
kubectl port-forward -n mlops svc/grafana 3000:80

# Run drift detection
python monitoring/drift_detection.py \
  --model-name customer-churn \
  --baseline-data data/baseline.csv \
  --current-data data/production.csv
```

## CI/CD Workflow

### Automated Testing

```bash
# Run all tests locally
poetry run pytest tests/ -v --cov=ml --cov=serving

# Run model validation
poetry run python tests/model_tests/validate_model.py

# Lint Dockerfiles
hadolint Dockerfile.train Dockerfile.serve
```

### GitHub Actions Workflows

**On Pull Request:**
1. Code quality checks (black, flake8, mypy)
2. Unit tests with coverage (>80%)
3. Integration tests
4. Model performance tests
5. Security scanning

**On Merge to Main:**
1. Build Docker images
2. Push to ECR
3. Deploy to staging environment
4. Run smoke tests
5. Promote to production (manual approval)

**Scheduled (Daily):**
1. Retrain models on new data
2. Evaluate model performance
3. Update model registry
4. Auto-deploy if performance improves

## MLflow Usage Examples

### Logging Experiments

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier

# Start MLflow run
with mlflow.start_run(run_name="rf-experiment"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Hyperparameter Tuning with Optuna + MLflow

```python
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3)
    }

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        return accuracy

study = optuna.create_study(direction="maximize")
mlflc = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="accuracy")
study.optimize(objective, n_trials=50, callbacks=[mlflc])
```

### Model Registry Operations

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = f"runs:/{run_id}/model"
mv = client.create_model_version(
    name="customer-churn",
    source=model_uri,
    run_id=run_id
)

# Transition to production
client.transition_model_version_stage(
    name="customer-churn",
    version=mv.version,
    stage="Production"
)
```

## Agile Workflow

**Sprint Planning**
- 2-week sprints
- User story grooming
- Task estimation with story points

**Daily Stand-ups**
- Automated via Slack bot
- MLflow experiment progress tracking
- Blocker identification

**Sprint Review**
- Model performance demos
- Experiment result presentations
- Stakeholder feedback

**Retrospectives**
- CI/CD pipeline improvements
- Model performance analysis
- Infrastructure optimization

**Tools**
- Jira for task tracking
- Confluence for documentation
- Slack for communication
- GitHub Projects for kanban boards

## Monitoring Dashboards

### Grafana Dashboards

**Model Performance Dashboard**
- Predictions per second
- Average latency (p50, p95, p99)
- Error rate by model version
- Model accuracy over time

**Infrastructure Dashboard**
- Pod resource usage
- Node availability
- Request queue depth
- Auto-scaling events

**Business Metrics Dashboard**
- Model prediction distribution
- Feature importance trends
- Cost per prediction
- ROI tracking

## Cost Optimization

**Implemented Strategies**
- Spot instances for training jobs
- Auto-scaling for model serving
- Model compression (ONNX, quantization)
- Efficient batch sizes
- Resource limits on pods
- S3 lifecycle policies
- Redshift workload management

**Expected Monthly Costs** (moderate usage)
- EKS cluster: ~$150
- RDS (MLflow): ~$50
- S3 storage: ~$30
- Redshift: ~$180 (dc2.large, 1 node)
- Data transfer: ~$20
- **Total: ~$430/month**

## Security

**Implemented Measures**
- IAM roles for service accounts (IRSA)
- Secrets stored in AWS Secrets Manager
- Network policies in Kubernetes
- TLS for all communications
- Container image scanning
- RBAC for Kubernetes access
- MLflow authentication enabled
- Audit logging to CloudWatch

## Best Practices Demonstrated

**MLOps**
- Experiment tracking and reproducibility
- Model versioning and lineage
- Automated model validation
- Continuous training pipelines
- A/B testing infrastructure

**DevOps**
- Infrastructure as Code (Terraform)
- Containerization (Docker)
- Orchestration (Kubernetes)
- CI/CD automation (GitHub Actions)
- Monitoring and observability

**Agile**
- Sprint-based development
- Iterative model improvements
- Stakeholder collaboration
- Continuous feedback loops
- Documentation as code

## Documentation

- [Architecture Deep Dive](docs/architecture.md)
- [MLflow Complete Guide](docs/mlflow_guide.md)
- [Deployment Guide](docs/deployment.md)
- [Monitoring & Alerting](docs/monitoring.md)

## License

This project is part of a professional portfolio and is available for review and evaluation purposes.

## Contact

Eduardo Peiter
- GitHub: @Dudomon
- Location: Brazil
- Languages: Portuguese (Native), English (Professional)

## Acknowledgments

This platform demonstrates production-ready MLOps practices used by leading tech companies including experiment tracking, automated deployment, and comprehensive monitoring.

---

**Key Skills Demonstrated:** MLOps, MLflow, AWS (EKS, RDS, S3, Redshift), Kubernetes, Helm, CI/CD (GitHub Actions), Docker, Terraform, Python, FastAPI, Prometheus, Grafana, Agile Methodologies

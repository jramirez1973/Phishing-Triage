# Phishing Triage System

An automated phishing detection and enrichment service that combines machine learning, threat intelligence, and sandbox analysis to provide comprehensive phishing risk assessment.

## 🚀 Features

- **Machine Learning Classification**: Advanced URL feature extraction and classification using scikit-learn
- **Threat Intelligence Integration**: Real-time URLhaus lookups for known malicious URLs
- **Sandbox Detonation**: Optional URL analysis via ANY.RUN or Joe Sandbox
- **Email Analysis**: Parse `.eml` files and extract URLs for analysis
- **Drift Detection**: Automated monitoring for model drift using ADWIN
- **Comprehensive Reports**: Detailed markdown reports with IOCs and recommendations
- **RESTful API**: FastAPI-based service with automatic documentation
- **MLflow Integration**: Model versioning and experiment tracking

## 📋 Requirements

- Python 3.11+
- API keys for enrichment services (optional):
  - URLhaus API key
  - ANY.RUN API key
  - Joe Sandbox API key

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/phish-triage.git
   cd phish-triage
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Initialize the database**
   ```bash
   python -c "from api.models import init_db; init_db()"
   ```

## 🎯 Quick Start

### 1. Train the Model

First, you need to train the phishing detection model:

```bash
# Download dataset (PhiUSIIL recommended)
# Place in data/phiusiil.csv

# Train model
python -m ml.train
```

The training script will:
- Extract features from URLs
- Train a Gradient Boosting classifier
- Log metrics to MLflow
- Save the model to `ml/model.joblib`

### 2. Start the Service

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`

### 3. Submit URLs for Analysis

```bash
# Submit a URL
curl -X POST http://localhost:8000/submit \
  -H "Content-Type: application/json" \
  -d '{"url": "http://suspicious-site.com/login", "detonate": false}'

# Get the report
curl http://localhost:8000/report/{submission_id}
```

## 🔧 API Endpoints

### Core Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /submit` - Submit URL or email for analysis
- `GET /report/{id}` - Get analysis report
- `GET /metrics` - Service metrics

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📊 Model Training

### Dataset

The system is designed to work with the [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (2024) which contains ~235k URLs with modern features.

### Features

The model extracts 35+ features from URLs including:
- URL structure metrics (length, components)
- Domain characteristics (entropy, TLD analysis)
- Suspicious token detection
- Protocol and port analysis
- Character distribution ratios

### Training Process

```python
# Basic training
python -m ml.train

# View MLflow UI
mlflow ui
```

### Model Performance

Expected performance metrics:
- ROC-AUC: ~0.95
- Precision: ~0.90
- Recall: ~0.85

## 🔍 Enrichment Services

### URLhaus Integration

[URLhaus API Documentation](https://urlhaus-api.abuse.ch/)

```python
# Automatic lookup for all submitted URLs
# Results included in risk assessment
```

### ANY.RUN Sandbox

[ANY.RUN API Documentation](https://any.run/api-documentation/)

```python
# Enable detonation in submission
{
  "url": "http://suspicious.com",
  "detonate": true,
  "provider": "anyrun"
}
```

### Joe Sandbox

[Joe Sandbox Integration](https://github.com/joesecurity/jbxapi)

```python
# Enable detonation in submission
{
  "url": "http://suspicious.com",
  "detonate": true,
  "provider": "joe"
}
```

## 📈 Monitoring & Drift Detection

### Automated Drift Detection

The system uses [River's ADWIN](https://riverml.xyz/dev/api/drift/ADWIN/) algorithm to detect distribution drift:

```bash
# Run drift check manually
python -m ml.drift

# Schedule as cron job
0 */6 * * * cd /path/to/phish-triage && python -m ml.drift
```

### Metrics Endpoint

Monitor system health and performance:

```bash
curl http://localhost:8000/metrics
```

## 📝 Report Format

Reports are generated in markdown format with:
- Executive summary with risk assessment
- Machine learning analysis results
- Threat intelligence findings
- Sandbox analysis (if performed)
- Extracted IOCs (URLs, IPs, domains, hashes)
- Recommended actions
- Technical details

## 🏗️ Architecture

```
phish-triage/
├── api/            # FastAPI application
│   ├── main.py     # API endpoints
│   ├── models.py   # Database models
│   ├── schemas.py  # Pydantic schemas
│   └── pipeline.py # Processing pipeline
├── ml/             # Machine learning
│   ├── train.py    # Model training
│   ├── features.py # Feature extraction
│   ├── predict.py  # Inference
│   └── drift.py    # Drift detection
├── enrich/         # External enrichment
│   ├── urlhaus.py  # URLhaus client
│   ├── anyrun.py   # ANY.RUN client
│   └── joesandbox.py # Joe Sandbox client
├── reports/        # Report generation
│   ├── render.py   # Report builder
│   └── templates/  # Jinja2 templates
└── storage/        # Data storage
    └── submissions.db # SQLite database
```

## 🔒 Security Considerations

1. **API Keys**: Store securely in environment variables
2. **Sandbox Safety**: Never execute malware locally
3. **Rate Limiting**: Respect third-party API limits
4. **Data Privacy**: Consider data retention policies
5. **Access Control**: Implement authentication for production

## 🧪 Testing

```bash
# Run unit tests
pytest

# Test with sample data
python -m ml.train  # Creates sample dataset if none exists

# Test API endpoints
python test_api.py
```

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [URLhaus API](https://urlhaus-api.abuse.ch/)
- [ANY.RUN API](https://any.run/api-documentation/)
- [Joe Sandbox API](https://www.joesecurity.org/joe-sandbox-api)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/quickstart/)
- [River ADWIN](https://riverml.xyz/dev/api/drift/ADWIN/)
- [PhiUSIIL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is designed for security research and defensive purposes. Always obtain proper authorization before analyzing URLs or files. The authors are not responsible for misuse of this tool.

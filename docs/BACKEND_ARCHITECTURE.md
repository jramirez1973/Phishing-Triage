# Backend Architecture Deep Dive

This document provides a detailed breakdown of the backend-end codebase, explaining the purpose and responsibility of each file and module.

---

### Project Structure Overview

The backend is organized into a modular structure to separate concerns, making the codebase easier to maintain and understand.

```
phish-triage/
└── backend/
    ├── api/
    │   ├── __init__.py
    │   ├── main.py         # FastAPI Endpoints & App Entrypoint
    │   ├── models.py       # SQLAlchemy Database Models
    │   ├── pipeline.py     # Core Analysis Orchestration
    │   └── schemas.py      # Pydantic Data Schemas
    ├── enrich/
    │   ├── __init__.py
    │   ├── advanced_intel.py # Threat Intelligence Aggregator
    │   └── urlhaus.py        # URLhaus API Client
    ├── ml/
    │   ├── __init__.py
    │   ├── features.py     # URL Feature Extraction Logic
    │   ├── model.joblib    # Pre-trained ML Model File
    │   ├── predict.py      # Model Loading & Prediction Logic
    │   └── train.py        # Script for Training the Model
    ├── reports/
    │   ├── __init__.py
    │   ├── openai_enhancer.py # OpenAI Report Summarization
    │   ├── render.py          # Markdown Report Generation
    │   └── templates/
    │       └── report.md.j2   # Jinja2 Report Template
    └── requirements.txt         # Python Dependencies
```

---

### Core Modules Explained

#### 1. `api/` - The Web Layer

This directory contains the core FastAPI application logic, handling all incoming HTTP requests and routing them to the correct services.

*   **`main.py`**: This is the main entry point of the application. It is responsible for:
    *   Initializing the FastAPI application.
    *   Configuring middleware, such as CORS for frontend communication.
    *   Defining all API endpoints (e.g., `/submit-url`, `/report/{id}`, `/health`).
    *   Handling the request/response lifecycle, including data validation using Pydantic schemas.
    *   Triggering the analysis pipeline for new submissions.
    *   Loading environment variables from the `.env` file.

*   **`pipeline.py`**: This is the heart of the analysis orchestration. The `handle_url_submission` function acts as a conductor, guiding a submission through every stage of the analysis in the correct order:
    1.  Score the URL with the ML model (`ml.predict`).
    2.  Enrich the data with external threat intelligence (`enrich.advanced_intel`).
    3.  Extract Indicators of Compromise (IOCs).
    4.  Generate an AI-powered executive summary (`reports.openai_enhancer`).
    5.  Compile all findings into a final report (`reports.render`).

*   **`models.py`**: Defines the application's database structure using SQLAlchemy ORM.
    *   The primary model, `Submission`, maps to the `submissions` table and holds all data for a single analysis task, including the input URL, status, final score, and the generated report.
    *   It also contains helper functions for database initialization (`init_db`) and session management (`get_db`).

*   **`schemas.py`**: Contains all Pydantic models. These schemas are used by FastAPI to:
    *   Validate the structure and data types of incoming request bodies.
    *   Serialize data into a consistent JSON format for outgoing responses.
    *   This ensures data integrity throughout the API.

#### 2. `ml/` - The Machine Learning Engine

This module contains everything related to the phishing prediction model.

*   **`train.py`**: A command-line script used for training, evaluating, and saving the machine learning model. It uses `MLflow` to log experiments, track model performance metrics, and register model versions.
*   **`predict.py`**: Provides the `score_url` function, which is the primary interface for the rest of the application to get a risk score for a URL. It handles loading the pre-trained `model.joblib` file and making predictions.
*   **`features.py`**: Contains the detailed logic for feature extraction. The `url_features` function takes a raw URL and converts it into a numerical feature vector that the machine learning model can understand.
*   **`model.joblib`**: This is not code, but the final output of the training process. It is the serialized (pickled) and trained scikit-learn classifier, ready to be loaded for inference.

#### 3. `enrich/` - External Intelligence

This module is responsible for gathering additional context on a URL from third-party threat intelligence providers.

*   **`advanced_intel.py`**: Implements the `ThreatIntelAggregator`, a class designed to manage and query multiple intelligence sources.
*   **`urlhaus.py`**: Contains the specific client logic for querying the Abuse.ch URLhaus API.

#### 4. `reports/` - Report Generation

This module compiles all the collected data into a final, human-readable report.

*   **`openai_enhancer.py`**: Contains the `enhance_report_with_openai` function. It constructs a detailed prompt with the analysis data and sends it to the OpenAI API (`gpt-4o-mini`) to receive a high-quality, AI-generated executive summary.
*   **`render.py`**: The `build_report` function uses the Jinja2 templating engine to populate a template with all the analysis results (ML score, IOCs, AI summary, etc.).
*   **`templates/report.md.j2`**: A Jinja2 template file that defines the structure and layout of the final Markdown report. This allows the report's format to be easily changed without modifying Python code.

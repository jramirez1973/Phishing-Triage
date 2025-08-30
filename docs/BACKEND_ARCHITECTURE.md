# Backend Architecture: A Comprehensive File-by-File Analysis

This document provides a definitive, in-depth analysis of every file and directory within the backend codebase. It is designed to be a technical reference for understanding the specific responsibilities and interactions of each component.

---

## Directory 1: `api/` - The Application Core & Web Layer

This directory is the heart of the application. It handles incoming web requests, manages the database, orchestrates the analysis process, and defines the data structures for communication.

### ➤ `api/main.py`
*   **Primary Responsibility**: **API Server Entrypoint**. This file defines and runs the main FastAPI application.
*   **Key Components**:
    *   **FastAPI App Initialization**: `app = FastAPI(...)` creates the central application instance.
    *   **CORS Middleware**: `app.add_middleware(CORSMiddleware, ...)` is configured here to allow the frontend web page (running on a different port) to make requests to this backend server.
    *   **Startup Event**: The `@app.on_event("startup")` decorator registers the `init_db()` function to run once when the server starts, ensuring all necessary database tables are created.
    *   **API Endpoints (`@app.post`, `@app.get`)**:
        *   `/submit-url`: The primary endpoint for receiving new URLs. It validates the input using the `SubmitURL` schema, creates an initial record in the database, and then hands off processing to `pipeline.py`. After the pipeline finishes, it updates the database record with the results.
        *   `/report/{submission_id}`: Allows a client to retrieve a completed analysis report by its unique ID. It queries the database and returns the stored data.
        *   `/health`: A simple endpoint used by the deployment environment (Render) to verify that the application is running and healthy.
        *   `/metrics`: Provides a high-level overview of system usage statistics.
    *   **Static File Mounting**: `app.mount("/", StaticFiles(directory="frontend", html=True), ...)` is the crucial line that tells FastAPI to also act as a web server for the frontend's `index.html`, CSS, and JavaScript files. It is placed at the end to ensure it doesn't override the API endpoints.

### ➤ `api/pipeline.py`
*   **Primary Responsibility**: **Analysis Orchestrator**. This is arguably the most important file for understanding the application's workflow. It doesn't handle any web requests directly; instead, it orchestrates the step-by-step process of analyzing a submission.
*   **Key Functions**:
    *   **`handle_url_submission(...)`**: This is the main function. It takes a URL and executes the entire analysis sequence in a specific order:
        1.  Calls `ml.predict.score_url()` to get the machine learning risk score.
        2.  Calls `enrich.advanced_intel.ThreatIntelAggregator` to gather data from external sources like URLhaus.
        3.  Calls its own `extract_iocs()` helper to pull out indicators of compromise.
        4.  Calls `reports.openai_enhancer.enhance_report_with_openai()` to generate a summary.
        5.  Calls `reports.render.build_report()` to assemble the final Markdown report.
        6.  Returns all of these artifacts back to `main.py`.

### ➤ `api/models.py`
*   **Primary Responsibility**: **Database Schema Definition**. This file defines the structure of the application's database tables using SQLAlchemy's ORM (Object-Relational Mapping).
*   **Key Components**:
    *   **`Submission` Class**: This class maps directly to the `submissions` table in the database. Each attribute of the class (e.g., `id`, `url`, `score`, `report_markdown`) corresponds to a column in that table. SQLAlchemy uses this model to translate Python objects into SQL queries.
    *   **`init_db()` Function**: Contains the logic to create the database file and the `submissions` table if they don't already exist.

### ➤ `api/schemas.py`
*   **Primary Responsibility**: **Data Validation and Serialization**. This file defines the expected shape of API request and response data using Pydantic.
*   **Key Components**:
    *   **`SubmitURL`**: Defines that an incoming submission request *must* contain a `url` field that is a valid URL.
    *   **`ReportResponse`**, **`SubmissionResponse`**, etc.: These models define the exact fields and data types that the API will send back in its JSON responses. This ensures consistency and helps auto-generate API documentation.

---
## Directory 2: `ml/` - The Machine Learning Engine

This directory contains all code and assets related to the phishing prediction model.

### ➤ `ml/train.py`
*   **Primary Responsibility**: **Model Training**. This is a standalone script that is run offline to create the `model.joblib` file. It is not part of the live API.
*   **Workflow**:
    1.  Loads a dataset of labeled phishing and legitimate URLs.
    2.  Uses `features.py` to convert each URL into a feature vector.
    3.  Splits the data into training and testing sets.
    4.  Trains a `GradientBoostingClassifier`.
    5.  Evaluates the model's performance.
    6.  Uses `joblib.dump()` to save the trained model object to the `model.joblib` file.

### ➤ `ml/predict.py`
*   **Primary Responsibility**: **Model Inference**. This module is the bridge between the live API and the offline-trained model.
*   **Key Functions**:
    *   **`load_model()`**: Caches the `model.joblib` file in memory so it doesn't have to be re-read from disk for every prediction, which improves performance.
    *   **`score_url(url)`**: This is the primary function used by the `pipeline.py`. It takes a single URL, calls `features.py` to generate the feature vector, and uses the loaded model to predict the phishing probability.

### ➤ `ml/features.py`
*   **Primary Responsibility**: **Feature Engineering**. This module contains the domain-specific logic for converting a URL string into a set of meaningful numerical features that a machine learning model can understand.
*   **Key Features Extracted**: It calculates dozens of features, including lexical features (`url_len`, `domain_entropy`, `special_char_ratio`), host-based features (`tldextract` is used here), and keyword-based features (presence of terms like "login", "secure", "verify").

---
## Directory 3: `enrich/` - External Data Enrichment

This directory contains the clients used to query external threat intelligence services.

### ➤ `enrich/advanced_intel.py`
*   **Primary Responsibility**: **Intelligence Aggregation**. This file's `ThreatIntelAggregator` class is designed to be a central point for managing multiple intelligence sources. Currently, it focuses on URLhaus but is structured to easily add more sources (e.g., VirusTotal, PhishTank) in the future.

### ➤ `enrich/urlhaus.py`
*   **Primary Responsibility**: **URLhaus API Client**. Contains the specific function `lookup_url` that knows how to construct the correct API request for the Abuse.ch URLhaus service, send it, and parse the JSON response.

---
## Directory 4: `reports/` - Report Generation & Formatting

This directory is responsible for compiling all the collected analysis data into a final, human-readable format.

### ➤ `reports/render.py`
*   **Primary Responsibility**: **Report Assembly**. This is the final step in the data-processing chain.
*   **Key Function**:
    *   **`build_report(...)`**: This function is the main workhorse. It receives all the data points from the pipeline (the URL, the ML score, the threat intel results, the IOCs, the AI summary). It then organizes this data into a "context" dictionary and uses the Jinja2 templating engine to pass this context to a template file.

### ➤ `reports/openai_enhancer.py`
*   **Primary Responsibility**: **AI-Powered Summarization**. This module integrates with the OpenAI API.
*   **Key Function**:
    *   **`enhance_report_with_openai(...)`**: Takes the raw JSON data from the analysis, formats it into a clear and concise prompt, and sends it to the `gpt-4o-mini` model. The prompt instructs the AI to act as a Tier 3 SOC analyst and provide an executive summary. The natural language text returned by the API is then used in the final report.

### ➤ `reports/templates/report.md.j2`
*   **Primary Responsibility**: **Report Structure and Layout**. This is not a Python file, but a Jinja2 template. It defines the structure of the final output report in Markdown format. It contains placeholders (like `{{ score }}`, `{{ summary }}`, `{{ iocs.ips }}`) that the `render.py` script fills in with the actual analysis data. This separation of logic and presentation makes it easy to change the report's layout without touching any Python code.

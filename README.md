# WalletAnalyser AI

WalletAnalyser AI is the intelligence layer of the WalletAnalyser platform.  
It is a dedicated **Python FastAPI** microservice, intentionally decoupled from the main backend, responsible for all AI, machine learning, and deep learning workloads.

While the backend handles portfolio data, authentication, and business logic, this service focuses exclusively on extracting insights, running predictive models, and delivering intelligent signals to enrich the user experience.

---

## 🧠 Purpose of the AI Service

This microservice is designed to power the analytical intelligence of WalletAnalyser:

- **Market sentiment analysis** — analyze the global sentiment (positive, neutral, pessimistic) around a stock by processing financial news, press releases, and price movements
- **Price & trend forecasting** — predict short and medium-term price movements for assets using time series models and deep learning architectures
- **Portfolio intelligence** — provide users with AI-driven insights on their portfolio: risk exposure, rebalancing suggestions, and asset positioning recommendations
- **Asset scoring** — score and rank assets based on momentum, sentiment, and historical patterns to help users make informed investment decisions
- **Similar portfolio clustering** — group portfolios by profile and surface comparable strategies or benchmarks

This service is designed to evolve alongside the platform, integrating new models and data sources as the product matures.

---

## 🛠️ Tech Stack

- **Python 3.13**
- **FastAPI**
- **Uvicorn**
- **uv** (dependency management)
- **Ruff** (linting & formatting)
- **Mypy** (static type checking)
- **Docker**
- **Azure App Service**
- **Azure Container Registry (ACR)**
- **GitHub Actions** (CI/CD)
- **Terraform** (infrastructure provisioning)

---

## 📦 Running the Project Locally

### Prerequisites

**Python 3.13** — install via the official pkg if not already installed:

```bash
python3 --version  # should print 3.13.x
```

**uv** — the package manager used for this project (equivalent of npm for Python):

```bash
pip3 install uv
```

---

### Setup & Start

Sync the project and install all dependencies from `pyproject.toml`:

```bash
uv sync
```

Start the development server:

```bash
uv run uvicorn app.main:app --reload --port 8080
```

The API will be available at:

```
http://localhost:8080
```

Interactive docs (Swagger UI):

```
http://localhost:8080/docs
```

---

## 🔧 Environment Variables

Create a `.env` (take `.env.example` for example) file at the project root for local development (this file is gitignored):

```
PORT=8080
DATABASE_URL=xxxx
```

---

## 📦 Dependency Management

This project uses **uv** to manage dependencies. Never edit `pyproject.toml` or `requirements.txt` manually — always go through `uv` commands.

Add a production dependency:

```bash
uv add <package>
```

Add a development dependency:

```bash
uv add --dev <package>
```

Remove a dependency:

```bash
uv remove <package>
```

`pyproject.toml` and `uv.lock` are updated automatically after every `uv add` or `uv remove`. Commit both files — `uv.lock` is the source of truth for reproducible installs across environments.

After any dependency change, regenerate the requirements files used by Docker:

```bash
uv export --no-dev -o requirements.txt
uv export -o requirements-dev.txt
```

---

## 🧹 Code Linting & Formatting

**Ruff** handles both linting and formatting in a single tool — the Python equivalent of ESLint + Prettier combined. Configuration lives in `.ruff.toml`.

Run the linter:

```bash
uv run ruff check .
```

Auto-fix linting issues:

```bash
uv run ruff check . --fix
```

Format the code:

```bash
uv run ruff format .
```

Run static type checking:

```bash
uv run mypy app/
```

---

## 🐳 Docker Support

Build the Docker image:

```bash
docker build -t walletanalyser-ai .
```

Run the container:

```bash
docker run -p 8080:8080 walletanalyser-ai
```

---

## ☁️ Azure Deployment (CI/CD)

A GitHub Actions pipeline automatically:

1. Checks out the repository
2. Logs into Azure
3. Builds the Docker image via `az acr build` and pushes it to the Azure Container Registry
4. Restarts the Azure App Service to pull and run the new image

Terraform ensures the App Service is always configured to pull the latest image from ACR using a system-assigned managed identity.

---

## 📄 Available Commands

`uv sync` — install all dependencies  
`uv add <package>` — add a production dependency  
`uv add --dev <package>` — add a development dependency  
`uv remove <package>` — remove a dependency  
`uv run uvicorn app.main:app --port 3001 --reload` — start dev server  
`uv run ruff check .` — lint  
`uv run ruff format .` — format  
`uv run mypy app/` — type check

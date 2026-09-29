# MLOps Assignment: GitHub Actions CI/CD + Airflow DAG

A small Iris-classifier project with:

1. **CI/CD (GitHub Actions)** – lint + test on PR, train on push to `main`, deploy to staging on tag.
2. **Airflow DAG** – download → preprocess → train → evaluate → deploy **only if accuracy improves**.

## Project structure

```
ml-cicd-airflow/
├── .github/workflows/
│   ├── pr.yml               # lint + tests on pull request
│   ├── train.yml            # train model on push to main
│   └── deploy-staging.yml   # deploy to staging on tag v*
├── airflow/dags/ml_pipeline_dag.py   # Airflow DAG
├── src/mlpipeline/          # data, preprocess, train, evaluate, deploy, serve, cli
├── tests/                   # pytest tests
├── docker-compose.yml       # runs Airflow locally
├── Dockerfile               # staging image (FastAPI model server)
├── requirements.txt / requirements-dev.txt
├── Makefile
└── docs/screenshots/        # put your screenshots here
```

---

## Part 0 – Run locally first (sanity check)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Tests + lint
PYTHONPATH=src pytest --cov=mlpipeline
flake8 src tests airflow
black --check src tests airflow

# Run the full pipeline once
PYTHONPATH=src python -m mlpipeline.cli all
```
(Windows PowerShell: `$env:PYTHONPATH="src"` before the commands.)

Output: `models/model.joblib`, `models/metrics.json`, and (if promoted) `models/production/`.

---

## Part 1 – GitHub Actions CI/CD

### Step 1: Create the GitHub repo and push the code
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
Pushing to `main` triggers **train.yml** (workflow #2).

### Step 2: Enable permissions (one time)
* Repo → **Settings → Actions → General → Workflow permissions** → select **Read and write permissions**
  (needed so the staging job can push the Docker image to GHCR).
* Repo → **Settings → Environments → New environment** → name it `staging` (used by deploy workflow).

### Step 3: Test the PR workflow (`pr.yml`) – lint + tests
```bash
git checkout -b feature/test-ci
echo "# change" >> README.md
git add . && git commit -m "Test PR workflow"
git push -u origin feature/test-ci
```
Open a Pull Request into `main` on GitHub → the **PR - Lint & Test** workflow runs
(`lint` job, then `test` job on Python 3.10 and 3.11).

📸 **Screenshots to take**
* PR page showing the green checks
* Actions tab → *PR - Lint & Test* run (all jobs green)

> To also show a failing lint check, add an unused import to a file, push, screenshot the red ❌, then fix it.

### Step 4: Test training on push to main (`train.yml`)
Merge the PR (or push directly to `main`). Actions → **Train model (main)** runs the pipeline and uploads
`trained-model` as an artifact; metrics appear in the job summary.

📸 **Screenshots**: workflow run page (green), job summary with metrics, Artifacts section.

### Step 5: Deploy to staging on tag (`deploy-staging.yml`)
```bash
git checkout main && git pull
git tag v1.0.0
git push origin v1.0.0
```
This trains the model, builds the Docker image, pushes it to `ghcr.io/<user>/<repo>:staging`,
runs it, and smoke-tests `/health` and `/predict`.

📸 **Screenshots**: Actions run for the tag (green), the "Smoke test staging" step log,
Repo → **Packages** showing the image.

> Real server deploy: uncomment the `ssh ...` line in `deploy-staging.yml` and add
> `STAGING_HOST` / `STAGING_SSH_KEY` in Settings → Secrets.

### YAML files to submit
`.github/workflows/pr.yml`, `.github/workflows/train.yml`, `.github/workflows/deploy-staging.yml`

---

## Part 2 – Airflow DAG (tool-based)

DAG file: `airflow/dags/ml_pipeline_dag.py`

```
download_data >> preprocess_data >> train_model >> evaluate_model >> [deploy_model | skip_deployment]
```

* `download_data` – downloads the Iris CSV (falls back to scikit-learn's bundled copy if offline)
* `preprocess_data` – cleans data, stratified train/test split
* `train_model` – RandomForest; params read from Airflow Variables `n_estimators`, `random_seed`
* `evaluate_model` – computes test accuracy, **branches**: deploy if accuracy > current production accuracy
  (or if there is no production model yet), else skip
* `deploy_model` – promotes the model to `models/production/`

### Option A – Docker (recommended)

**Step 1:** Install Docker Desktop and make sure it is running.

**Step 2:** From the project root:
```bash
# Linux/macOS (fixes file permissions on the mounted folder)
export AIRFLOW_UID=$(id -u)
docker compose up -d
```
Windows: just `docker compose up -d`.

**Step 3:** Wait ~2 minutes (first start installs pandas/scikit-learn). Get the admin password:
```bash
docker compose logs airflow | grep -i password
```
Open **http://localhost:8080**, login as `admin` with that password.

**Step 4:** In the UI find DAG **ml_training_pipeline** → toggle it **ON** (unpause) → click ▶ **Trigger DAG**.

📸 **Screenshots to take (Airflow UI)**
1. DAG list showing `ml_training_pipeline`
2. **Graph** view of the DAG (structure)
3. **Grid** view after a successful run (green tasks; `deploy_model` green, `skip_deployment` pink/skipped)
4. Log of `evaluate_model` (shows accuracy)
5. **Admin → XCom** or task *XCom* tab with metrics

**Step 5 – Show the "only if accuracy improves" logic**
* First run → no production model yet → `deploy_model` runs.
* Trigger again without changes → same accuracy → `skip_deployment` runs (no improvement).
* Change hyper-parameters: **Admin → Variables → +** add `n_estimators` = `5` and `random_seed` = `7`,
  trigger again. If accuracy is better → deploy; if worse → skip. Take a screenshot of both outcomes.

Stop Airflow: `docker compose down`

### Option B – Without Docker (Linux/macOS/WSL)
```bash
python -m venv .venv-airflow && source .venv-airflow/bin/activate
pip install "apache-airflow==2.9.3" pandas scikit-learn joblib requests \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"
export AIRFLOW_HOME=$(pwd)/airflow_home
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export PYTHONPATH=$(pwd)/src
airflow standalone
```
Then open http://localhost:8080 and follow Steps 4–5 above.

---

## What to submit

| Item | Files |
|------|-------|
| GitHub Actions YAML | the 3 files in `.github/workflows/` |
| Pipeline run screenshots | `docs/screenshots/github_actions/` (PR checks, train run, staging deploy run) |
| Airflow DAG code | `airflow/dags/ml_pipeline_dag.py` |
| Airflow UI screenshots | `docs/screenshots/airflow/` (DAG list, graph, grid, logs, both branches) |

## Troubleshooting

* **`ModuleNotFoundError: mlpipeline`** → set `PYTHONPATH=src` (the DAG also adds it automatically).
* **Airflow DAG import error** → check the *DAG Import Errors* banner in the UI; make sure pip packages finished installing (`docker compose logs airflow`).
* **Permission denied writing `data/` or `models/` (Linux)** → `export AIRFLOW_UID=$(id -u)` before `docker compose up`.
* **GHCR push fails (403)** → enable *Read and write permissions* for workflows (Part 1, Step 2).
* **`black --check` fails** → run `black src tests airflow` and commit.

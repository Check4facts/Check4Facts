# Check4Facts — Machine Learning & Fact‑Checking Module

This repository contains the **Python machine‑learning back‑end module** of the
[Check4Facts](https://check4facts.gr/) platform — a research project that studies
fact‑checking in the Greek public sphere, focusing on the credibility of
statements made by public/political figures.

The platform combines automated Machine Learning (ML) techniques with the
expertise of human fact‑checkers to produce trustworthy, detailed assessment
reports. Reports are published to the public portal and currently revolve around
five topics: **Immigration, Crime, Climate Change, Pandemic, and Digital
Transition**.

Each statement is assessed on a five‑grade accuracy scale:

| Code | Label (EN)             | Label (GR)              |
|:----:|------------------------|-------------------------|
| `0`  | Unverifiable           | Μη επαληθεύσιμη          |
| `1`  | Inaccurate             | Ανακριβής               |
| `2`  | Relatively Inaccurate  | Σχετικά ανακριβής        |
| `3`  | Relatively Accurate    | Σχετικά ακριβής          |
| `4`  | Accurate               | Ακριβής                 |

> This module is **only the ML / fact‑checking back‑end**. The public website and
> the fact‑checker web app (React front‑end + Spring Boot back‑end) live in
> separate repositories and talk to this module over its REST API. This module
> reads from / writes to the shared PostgreSQL database and uses Redis + Celery
> for asynchronous task execution.

---

## Table of Contents

1. [What this module does](#what-this-module-does)
2. [Architecture overview](#architecture-overview)
3. [Machine Learning tools offered](#machine-learning-tools-offered)
   - [1. Classic credibility analysis pipeline](#1-classic-credibility-analysis-pipeline)
   - [2. Article summarization (LLM)](#2-article-summarization-llm)
   - [3. RAG justification (LLM + web search)](#3-rag-justification-llm--web-search)
4. [REST API reference](#rest-api-reference)
5. [WebSocket: real‑time task progress](#websocket-real-time-task-progress)
6. [Project structure](#project-structure)
7. [Setup & installation](#setup--installation)
8. [Configuration](#configuration)
9. [Running the module](#running-the-module)
10. [CLI usage (offline / dev)](#cli-usage-offline--dev)
11. [Deployment notes](#deployment-notes)

---

## What this module does

Given a Greek statement, this module can:

- **Analyze** it with the classic ML pipeline — search the web for relevant
  resources, harvest them, extract linguistic/semantic features, and predict a
  credibility score using a trained scikit‑learn model.
- **Train / retrain** the credibility classifier on labelled statements stored in
  the database.
- **Summarize** a saved report/article into a short Greek bulleted list using LLMs.
- **Justify (RAG)** a statement — search the live web, crawl & embed the most
  relevant passages, and ask an LLM to produce a verdict (one of the five labels)
  together with a justification and the list of source URLs used.

All long‑running operations are executed asynchronously as **Celery tasks** and
their progress is streamed back to clients through **Redis Pub/Sub** and a
**WebSocket** endpoint.

---

## Architecture overview

```
                 ┌────────────────────────────────────────────────┐
   React /       │              FastAPI app (this repo)            │
   Spring Boot   │   check4facts/api/__init__.py                   │
   front-end ───▶│   • JWT-protected REST endpoints                │
   (JWT)         │   • WebSocket /ws/{task_id} for live progress   │
                 └───────────────┬────────────────────────────────┘
                                 │ apply_async()
                                 ▼
                 ┌────────────────────────────────────────────────┐
                 │            Celery workers (tasks.py)            │
                 │  analyze · train · summarize · justify (RAG)    │
                 └───┬───────────────┬───────────────┬────────────┘
                     │               │               │
            ┌────────▼─────┐  ┌──────▼───────┐  ┌────▼──────────────┐
            │ ML pipeline  │  │  LLM clients │  │  Web search /     │
            │ search→      │  │  Gemini/Groq │  │  crawl + embed    │
            │ harvest→     │  │  Mistral/    │  │  (crawl4ai,       │
            │ features→    │  │  Ollama      │  │  SearXNG, DDG,    │
            │ predict      │  │              │  │  Google CSE)      │
            └──────┬───────┘  └──────┬───────┘  └────┬──────────────┘
                   │                 │               │
                   ▼                 ▼               ▼
            ┌────────────────────────────────────────────────┐
            │      PostgreSQL (statements, features,          │
            │      resources, reports, summaries,             │
            │      justifications)   +   Redis (broker/PubSub)│
            └────────────────────────────────────────────────┘
```

- **API layer:** FastAPI (`check4facts/api/__init__.py`). Endpoints authenticate
  requests via JWT (HS512, base64‑encoded secret shared with the platform
  back‑end), enqueue Celery tasks, and return a `taskId`.
- **Task layer:** Celery (`check4facts/api/tasks.py`), backed by Redis.
- **Progress:** Each task publishes JSON progress messages to a Redis channel;
  the WebSocket endpoint relays them to the connected client.
- **Storage:** PostgreSQL via `check4facts/database.py` (`DBHandler`).

---

## Machine Learning tools offered

### 1. Classic credibility analysis pipeline

The original Check4Facts ML workflow. It does **not** use LLMs; it relies on
classic NLP features and a trained scikit‑learn classifier.

Pipeline stages (`check4facts/scripts/` + `train.py`/`predict.py`):

1. **Search** (`scripts/search.py`, `SearchEngine`) — converts a statement into a
   keyword query and fires it against the **Google Custom Search JSON API** to get
   related web results. Configured via `config/search_config.yml`.
2. **Harvest** (`scripts/harvest.py`, `Harvester`) — scrapes each result URL
   (BeautifulSoup / lxml), extracts clean text, and for every resource isolates
   the **title**, **body**, **most similar paragraph** and **most similar
   sentence** w.r.t. the statement (BoW or embedding similarity).
   Configured via `config/harvest_config.yml`.
3. **Feature extraction** (`scripts/features.py`, `FeaturesExtractor`) — builds
   feature vectors capturing: spaCy **embeddings**, **similarity**,
   **subjectivity**, **sentiment**, **emotion** (anger/disgust/fear/happiness/
   sadness/surprise) and **polarity counts**, at multiple granularities
   (statement, resource title/body/similar‑paragraph/similar‑sentence).
   Uses spaCy (`el_core_news_lg`), NLTK, polyglot and a Greek sentiment lexicon
   (`greek_sentiment_lexicon.tsv`). Configured via `config/features_config.yml`.
4. **Train** (`train.py`, `Trainer`) — runs a cross‑validated grid search over a
   configurable set of classifiers (Naive Bayes, kNN, Logistic Regression, SVM,
   MLP, Decision Tree, Random Forest, Extra Trees) and saves the best model as a
   `.joblib` file under `models/`. Configured via `config/train_config.yml`.
5. **Predict** (`predict.py`, `Predictor`) — loads a saved model and outputs a
   credibility prediction with confidence. The model path is set in
   `config/predict_config.yml` (the trained `.joblib` model is provided separately
   and is not tracked in the repo).

Triggered through the `/analyze`, `/train` and `/intial-train` endpoints.

### 2. Article summarization (LLM)

Generates a short Greek bulleted summary (3–4 bullets) of a saved report/article.
Code lives in `check4facts/scripts/text_sum/`.

Summarization uses a **fallback chain of models** so the feature keeps working
even if one provider is unavailable:

1. **Gemini** (`google-generativeai`, `gemini-2.0-flash`) — `local_llm.google_llm`
2. **Groq** (`langchain-groq`, two configurable models with retry/long‑text
   map‑reduce chunking) — `groq_api.groq_api`
3. **Local Hugging Face model** (`google-t5/t5-small` via `transformers`, with
   Greek↔English translation around it) — `local_llm.invoke_local_llm`

Article HTML is cleaned with `text_process.extract_text_from_html`, the summary is
converted to an HTML bullet list, and stored back to the DB
(`DBHandler.insert_summary`).

Triggered through `/summarize/{article_id}` and `/batch-summarize`.

### 3. RAG justification (LLM + web search)

Receives a statement, searches the live web, retrieves & embeds the most relevant
passages, and asks an LLM for a verdict + justification + supporting source URLs.

There are **two RAG implementations** in the repo:

- **Legacy RAG** — `check4facts/scripts/rag/pipeline.py` (`run_pipeline`).
  Searches with DuckDuckGo / Google CSE, harvests with a custom harvester, picks
  top‑n sources by body similarity, then runs the LLM fallback chain.
- **Current RAG (web crawler)** — `check4facts/scripts/web_crawler_rag/crawl4ai.py`
  (`crawl4ai`). This is what the `/justify` task uses today. It:
  1. Uses an LLM (`url_generation.py`) to generate ~10 targeted Greek/English
     search queries (with `site:` operators toward authoritative domains).
  2. Resolves queries to URLs via a self‑hosted **SearXNG** instance
     (`web_crawler_rag/search_engine.py`), with Google CSE / DDG fallbacks.
  3. Crawls pages asynchronously with **crawl4ai**, converts markdown→text,
     chunks it, and keeps chunks whose embedding cosine‑similarity to the claim
     exceeds a threshold.
  4. Computes embeddings either via **Ollama** (`OllamaEmbeddings.py`), a
     Hugging Face SentenceTransformer (`lighteternal/stsb-xlm-r-greek-transfer`,
     enabled by `USE_HF`), or a remote embeddings API (`EMBEDDINGS_API_URL`).
  5. Feeds the aggregated evidence to the **LLM fallback chain**:
     **Gemini → Groq → Mistral → Ollama (local)**.

The LLM returns text in a fixed format (`Statement:` / `Result of the statement:`
/ `Justification:`); the result is parsed, the label mapped to the 0–4 scale
(`rag/translate.py:translate_label`), and stored via
`DBHandler.insert_justification` together with the timestamp, elapsed time, model
name and source URLs.

Triggered through `/justify`, `/batch-justify` and the test endpoints
(`/rag-test`, `/new-rag-test`, `/rag-batch`).

LLM providers used across summarization & RAG:

| Provider | Library                  | Notes                                  |
|----------|--------------------------|----------------------------------------|
| Gemini   | `google-generativeai`    | `gemini-2.0-flash`                      |
| Groq     | `langchain-groq`         | 2 configurable models w/ retry         |
| Mistral  | `mistralai` client       | RAG fallback                           |
| Ollama   | `ollama`                 | local/remote fallback + embeddings     |
| HF/T5    | `transformers`, `torch`  | local summarization fallback           |

---

## REST API reference

Base URL in production: `http://127.0.0.1:9090` (behind the platform reverse
proxy). **All endpoints require a valid JWT.** REST endpoints expect it in the
`Authorization: Bearer <token>` header, while the WebSocket endpoint expects it as
a `?token=<JWT>` query parameter (browsers can't set custom headers on the
WebSocket handshake). In both cases the token is verified with the HS512 algorithm
against the base64‑decoded `JWT_SECRET_KEY`.

Most endpoints return immediately with a `taskId`; clients then poll
`/task-status/{task_id}` or subscribe to `/ws/{task_id}` for progress.

### Analysis & training

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/analyze` | `{ "id": <int>, "text": "<statement>" }` | Run the classic search→harvest→features→predict pipeline for one statement and persist resources + features + prediction. |
| `POST` | `/train` | — | Retrain the credibility classifier on features already stored in the DB; saves the best model to `models/`. |
| `GET`  | `/intial-train` | — | Full cold‑start training: for every statement in the DB, run search/harvest/features, then train. |

### Task status

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `GET`  | `/task-status/{task_id}` | — | Status + info for a single Celery task. |
| `POST` | `/batch-task-status` | `[ { "id": "<task_id>" }, ... ]` | Status for many tasks at once. |
| `GET`  | `/fetch-active-tasks` | — | Status for all task ids tracked as active in the DB. |

### Summarization

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/summarize/{article_id}` | — | Summarize one article/report and store the summary. |
| `POST` | `/batch-summarize` | — | Summarize all published articles that have no summary yet. |

### RAG / justification

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/justify` | `{ "id": <int>, "n": <int> }` | Run the current (crawl4ai) RAG pipeline for a statement using up to `n` web sources; store the justification, label, model and sources. |
| `POST` | `/batch-justify` | `{ "n": <int> }` | Run RAG justification for all statements in the DB. |

### Test / debug endpoints

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/test/summarize` | `{ "article_id": <int>, "text": "<raw text>" }` | Summarize arbitrary text (no DB write). |
| `POST` | `/rag-test` | `{ "article_id": <int>, "text": "<claim>", "n": <int> }` | Run the **legacy** RAG pipeline. |
| `POST` | `/new-rag-test` | `{ "article_id": <int>, "text": "<claim>", "n": <int> }` | Run the **current** crawl4ai RAG pipeline. |
| `POST` | `/rag-batch` | — | Run the batch RAG test routine. |
| `POST` | `/start-dummy-task` | — | Start a dummy task that emits progress (connectivity test). |

#### Example

```bash
# Kick off an analysis
curl -X POST http://127.0.0.1:9090/analyze \
  -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{"id": 233, "text": "Το 80-85% είναι πλέον οικονομικοί μετανάστες."}'
# -> {"status":"PROGRESS","taskId":"<uuid>","taskInfo":{"current":1,"total":4,"type":"233"}}

# Poll status
curl http://127.0.0.1:9090/task-status/<uuid> -H "Authorization: Bearer $JWT"
```

---

## WebSocket: real‑time task progress

```
ws://127.0.0.1:9090/ws/{task_id}?token=<JWT>
```

- The JWT is passed as the `token` query parameter (validated on handshake).
- On connect, any stored progress for the task (`progress:{task_id}` in Redis) is
  flushed, then the socket subscribes to the task's Redis Pub/Sub channel.
- Progress messages are JSON, e.g.
  `{"taskId": "...", "progress": 60, "status": "PROGRESS", "type": "justify"}`.
  Terminal states are `status: "SUCCESS"` or `status: "FAILURE"`.

---

## Project structure

```
check4facts/
├── api/
│   ├── __init__.py        # FastAPI app: REST endpoints + WebSocket + JWT/CORS
│   ├── tasks.py           # Celery task definitions (analyze/train/summarize/justify/rag)
│   ├── celery_worker.py   # Celery app bootstrap
│   ├── redis_pubsub.py    # Redis publish/get helpers for progress
│   └── uwsgi.ini          # (legacy) uWSGI config
├── scripts/
│   ├── search.py          # SearchEngine (Google CSE)
│   ├── harvest.py         # Harvester (web scraping + similarity extraction)
│   ├── features.py        # FeaturesExtractor (NLP features)
│   ├── text_sum/          # LLM summarization (Gemini/Groq/local T5)
│   ├── rag/               # Legacy RAG pipeline + LLM clients + search
│   └── web_crawler_rag/   # Current RAG (crawl4ai + SearXNG + embeddings)
├── train.py               # Trainer (grid search over sklearn classifiers)
├── predict.py             # Predictor (loads .joblib model)
├── database.py            # DBHandler (PostgreSQL access) + Redis channel naming
├── config.py              # DirConf (paths)
├── logging.py             # logger
├── metrics.py             # evaluation metrics
└── cli.py                 # command-line interface for offline/dev runs
config/                    # YAML configs (search/harvest/features/train/predict/db)
greek_sentiment_lexicon.tsv
Pipfile / Pipfile.lock     # dependencies (pipenv, Python 3.10)
.env.example               # environment variable template
```

---

## Setup & installation

### Prerequisites

- **Python 3.10**
- **PostgreSQL** (the shared Check4Facts database)
- **Redis** (Celery broker + result backend + Pub/Sub)
- Optional, depending on which features you use:
  - A **SearXNG** instance (current RAG search)
  - An **Ollama** server (local LLM / embeddings fallback)
  - API keys for **Gemini**, **Groq**, **Mistral**, **Google Custom Search**

### Install dependencies

This project uses **pipenv** (Python 3.10):

```bash
pip install pipenv
pipenv install        # installs everything from the Pipfile
pipenv shell
```

After installing, download the required NLP models/resources:

```bash
# spaCy Greek model used by feature extraction
python -m spacy download el_core_news_lg
# NLTK tokenizers (also auto-downloaded at runtime by some scripts)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

> `crawl4ai` may require a one‑time browser setup (`crawl4ai-setup` / Playwright
> install) on the host running the RAG worker.

---

## Configuration

### Environment variables (`.env`)

Copy `.env.example` to `.env` and fill in the values:

| Variable | Purpose |
|----------|---------|
| `JWT_SECRET_KEY` | **Base64‑encoded** secret shared with the platform back‑end; used to validate HS512 JWTs. Required. |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` / `CELERY_REDIS_URL` | Redis URLs for Celery and Pub/Sub. |
| `GEMINI_API_KEY` | Google Gemini key (summarization + RAG). |
| `GROQ_API_KEY_1` / `GROQ_API_KEY_2` | Groq keys (two for redundancy). |
| `GROQ_LLM_MODEL_1` / `GROQ_LLM_MODEL_2` | Groq model names. |
| `MISTRAL_API_KEY` | Mistral key (RAG fallback). |
| `GOOGLE_SEARCH_KEY` / `GOOGLE_CX_KEY` | Google Custom Search API key + engine id. |
| `SEARXNG_HOST` | URL of the self‑hosted SearXNG instance (current RAG search). |
| `USE_HF` | If set, use Hugging Face SentenceTransformer for embeddings instead of Ollama. |
| `EMBEDDINGS_API_URL` | If set, compute embeddings via a remote embeddings service. |

> **Security note:** never commit real secrets. Some YAML files under `config/`
> currently contain hard‑coded API keys / search‑engine ids — move these to
> environment variables / untracked configs and rotate any exposed keys.

### YAML configs (`config/`)

| File | Used by | Purpose |
|------|---------|---------|
| `db_config.yml` | `DBHandler` | PostgreSQL connection (dbname/user/password/host/port). |
| `search_config.yml` | `SearchEngine` | Google CSE params (result counts, language, engine id). |
| `harvest_config.yml` | `Harvester` | Scraping headers, timeout, blacklist tags, similarity metric. |
| `features_config.yml` | `FeaturesExtractor` | spaCy model, lexicon, which features/granularities to compute. |
| `train_config.yml` | `Trainer` | Classifier list + hyperparameter grids, CV settings, feature columns. |
| `predict_config.yml` | `Predictor` | Path to the trained model + feature columns. |

---

## Running the module

You need three things running: **PostgreSQL**, **Redis**, and then this module's
**API server** + **Celery worker(s)**.

### Development

```bash
pipenv shell

# 1) Start the FastAPI app (ASGI via uvicorn)
uvicorn check4facts.api:app --host 127.0.0.1 --port 9090 --reload

# 2) In another shell, start a Celery worker
celery -A check4facts.api.celery_worker.celery_app worker --loglevel INFO --pool threads
```

---

## CLI usage (offline / dev)

`check4facts/cli.py` exposes the pipeline stages directly for development/testing
(uses the YAML configs under `config/`). Run with:

```bash
python -m check4facts.cli <command> [--settings <file.yml>]
```

| Command | Description |
|---------|-------------|
| `search` / `search_dev` | Run the search component. |
| `harvest` / `harvest_dev` | Run the harvesting component. |
| `features` / `features_dev` | Run feature extraction. |
| `predict_dev` | Run model prediction. |
| `train_dev` | Run model training. |
| `analyze_task_demo` | Full demo workflow: search → harvest → features → predict → store. |
| `train_task_demo` | Train from features already in the DB and save the best model. |
| `initial_train` | Cold‑start training over all statements in the DB. |

Example:

```bash
python -m check4facts.cli analyze_task_demo \
  --search_settings search_config.yml \
  --harvest_settings harvest_config.yml \
  --features_settings features_config.yml \
  --predict_settings predict_config.yml \
  --db_settings db_config.yml
```

---

## Deployment notes

In production the module is typically run as two long‑lived services (e.g. via
**systemd**):

- **API** — the FastAPI app served by uvicorn:
  `uvicorn check4facts.api:app --host 127.0.0.1 --port 9090 --workers 4`
- **Workers** — Celery workers, e.g. started with `celery multi` and configured
  via an environment file (`CELERYD_NODES`, pid/log files, log level).

> Note: the active serving path is uvicorn/ASGI. The legacy `check4facts/api/uwsgi.ini`
> still points to a Flask entrypoint (`module = check4facts.api:flask_app`) and is
> outdated — prefer the uvicorn command above.

Typical production layout: Nginx/Apache reverse proxy → uvicorn (port 9090) for
the REST/WebSocket API, with one or more Celery workers consuming from Redis, and
the platform's Spring Boot back‑end issuing JWT‑authenticated requests.

CORS is currently configured for `http://localhost:8080` and
`http://localhost:9000`; update the origins in `check4facts/api/__init__.py` for
your deployment.

---

# Module 5 plan: containerised app and AWS deployment

Written 2026-11-23. Module 5 milestone: Docker + AWS deployment, login, logging
(`docs/planning/milestones.md`). Decisions so far: cloud = **AWS** (free plan, **Sydney
`ap-southeast-2`**: the project account is locked to that region; the earlier Mumbai
`ap-south-1` note was wrong, corrected 2026-11-24); hosted LLM = **Groq** free API. Order: platform-independent work first
(steps 1-5 below), cloud only after the app runs in Docker locally.

## 1. What exists today

| Part | State |
|---|---|
| Pipeline (`src/datacopilot/pipeline/`) | Profiler, answerability classifier (`models/answerability.joblib`, 3 KB, committed), SQL generator, validator (sqlglot, SELECT only), executor (DuckDB for CSVs, read-only SQLite with timeout), runner (validate + one retry). Explainer is an empty stub. |
| LLM | `sql_generator.py` calls **Ollama only** (`qwen2.5-coder:7b`, temperature 0, `num_ctx` 8192) over HTTP. `.env.example` already has `LLM_PROVIDER` but nothing reads it. |
| API | `src/datacopilot/api/main.py`: FastAPI app with `GET /health` only. |
| UI | `src/datacopilot/ui/`: empty package. |
| Data | Four processed datasets in `data/processed/` (~250 MB: `olist.sqlite` 147 MB, IPL / F1 / crop CSVs), gitignored, rebuilt by `scripts/prepare_data.py`. `datacopilot.questions.connect(dataset)` opens one. |
| Docker | `docker/` holds only `.gitkeep`. Docker Desktop 29.9.3 (Compose v5.7.0) installed on the Mac; buildx can build `linux/amd64`. |
| Logging | Only `print` in scripts; nothing structured. |
| Tests | 346 pytest tests, no network (LLM calls are faked). |

## 2. What is missing

1. **Switchable LLM provider**: `LLM_PROVIDER=ollama | groq`, same prompt, same
   `extract_sql`, same validator and retry; Groq key only from the environment.
2. **Backend**: `POST /ask` (dataset + question -> gate decision, SQL, rows, timings),
   `GET /datasets`; dataset connections and schemas cached at startup.
3. **UI**: Streamlit app that calls the backend over HTTP (never imports the pipeline), so
   UI and API can be deployed and scaled separately.
4. **Containers**: Dockerfiles for API and UI, `docker-compose.yml`, amd64 builds.
5. **Structured logging**: one JSON line per request: request id, dataset, question, gate
   decision and probability, SQL, outcome, row count, latency per stage, provider/model.
   Never the API key or other secrets.
6. Later in Module 5 (cloud): data in S3, secrets in Secrets Manager, logs in CloudWatch,
   login, HTTPS.

## 3. Target architecture

```
 Browser
    |
    v
 [ui] Streamlit container  --HTTP-->  [api] FastAPI container
                                         |  1. answerability gate (XGBoost, in-process)
                                         |  2. LLM writes SQL  --> provider (env LLM_PROVIDER)
                                         |        ollama: host Ollama (local dev only)
                                         |        groq:   api.groq.com (cloud), key from env
                                         |  3. validator (sqlglot, SELECT only) + one retry
                                         |  4. executor (DuckDB / read-only SQLite)
                                         |  5. JSON log line to stdout
                                         v
                                   data/processed (volume locally; S3 -> container in cloud)
```

- **Same pipeline for both providers.** Only the transport changes (Ollama
  `/api/generate` vs Groq's OpenAI-compatible chat completions). Prompt, temperature 0,
  SQL extraction, validator and retry are shared, so Module 4 results stay comparable.
- **Local:** `docker compose up` runs `api` + `ui`; Ollama keeps running on the Mac (GPU)
  and the API reaches it at `http://host.docker.internal:11434`. Data is mounted read-only
  from `./data/processed`.
- **Cloud:** the same images with `LLM_PROVIDER=groq`. No Ollama in the cloud: a 7B model
  needs a GPU or far more memory than free-plan instances have.
- **Images:** `python:3.13-slim`, non-root user, no data and no `.env` inside the image;
  built for `linux/amd64` for the cloud (the Mac is ARM).
- **Logs** go to stdout as JSON; Docker keeps them locally, CloudWatch collects them in AWS.

## 4. AWS services (first guess; the deployment plan is `docs/notes/module5_aws_plan.md`)

| Need | Service | Notes |
|---|---|---|
| Container images | **ECR** (Elastic Container Registry) | Push the amd64 `api` and `ui` images. |
| Container hosting | **EC2** instance running Docker Compose (simplest; first choice), or **ECS on Fargate** | EC2: one small instance, the same `docker-compose.yml`. Memory matters: pandas + DuckDB + XGBoost + Streamlit will not fit comfortably in 1 GB, so a 2 GB instance (e.g. t3.small) is likely; check what the free plan / credits cover. Fargate: no server to manage, billed per vCPU-hour. |
| Datasets | **S3** | Upload `data/processed/` (~250 MB) once; the container syncs it at start into a local path (DuckDB / SQLite need local files). |
| Secrets | **Secrets Manager** (or SSM Parameter Store, which has a free standard tier) | `GROQ_API_KEY`; injected as an environment variable at start, never baked into images or committed. |
| Logs and metrics | **CloudWatch Logs** | awslogs driver (EC2) or the ECS log configuration; JSON lines are searchable with Logs Insights. A billing alarm too. |
| Permissions | **IAM** | Instance / task role with only: read the S3 data prefix, read the one secret, write logs, pull from ECR. No access keys on the server. Root account with MFA, a separate admin user for daily work. |
| Network | VPC default, security group | Open only the UI port (and SSH from your IP for EC2); HTTPS via a load balancer or a reverse proxy is a later step. |

Cost guardrails before creating anything: AWS Budgets alert, region fixed to `ap-southeast-2`,
stop or delete resources after demos. App Runner is out: an organization SCP denies it in this
account, and AWS closed it to new customers from 2026-04-30.

## 5. Steps

| # | Step | Done when |
|---|---|---|
| 1 | This plan | **Done** 2026-11-23 |
| 2 | Switchable LLM provider (`ollama` default, `groq`) | **Done**: `pipeline/llm.py`; Groq not yet live-tested (needs a key) |
| 3 | FastAPI `/health`, `/ask`, `/datasets`; Streamlit UI calling the API | **Done**: `service.py`, `api/main.py`, `ui/app.py` |
| 4 | Dockerfiles + `docker-compose.yml`; amd64 build | **Done**: `docker/Dockerfile` (targets `api`, `ui`); amd64 API image answered a question under emulation |
| 5 | Structured JSON request logging | **Done**: `request_log.py`; one line per `/ask`, secrets redacted |
| 6 | AWS: ECR, S3 data, secret, EC2, CloudWatch (plan: `module5_aws_plan.md`, waiting for approval) | public URL answers a question with Groq |
| 7 | Login (milestone item); HTTPS | options in `module5_aws_plan.md`; decision pending |

## 6. Running it locally

```bash
docker compose up --build            # API on :8000, UI on http://localhost:8501; Ollama on the host
docker compose logs -f api           # one JSON line per /ask
docker compose down

# images for the cloud (x86), built on the ARM Mac:
docker buildx build --platform linux/amd64 -f docker/Dockerfile --target api -t datacopilot-api:amd64 --load .
docker buildx build --platform linux/amd64 -f docker/Dockerfile --target ui  -t datacopilot-ui:amd64  --load .
```

Without Docker: `uvicorn datacopilot.api.main:app --app-dir src` and
`API_URL=http://localhost:8000 streamlit run src/datacopilot/ui/app.py`.

To use Groq: set `LLM_PROVIDER=groq`, `GROQ_API_KEY` and `GROQ_MODEL` in `.env` (Compose passes
only these settings into the API container). Images: API 848 MB (`xgboost-cpu`, no NVIDIA
libraries), UI 813 MB.

## 7. Open points

- **Groq model: `openai/gpt-oss-120b`** (decided 2026-11-23). Text models available to the key:
  `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`. Compared on 24 **train-split**
  questions (6 per dataset, 7 Hinglish; no test or held-out rows), same prompt and validator:

  | Model | Strict | Relaxed | Failed SQL |
  |---|---|---|---|
  | openai/gpt-oss-20b | 12/24 | 17/24 | 0 |
  | openai/gpt-oss-120b | 11/24 | 17/24 | 0 |
  | qwen/qwen3.8-27b | 10/24 | 16/24 | 0 |
  | Ollama qwen2.5-coder:7b (Module 4) | 10/24 | 11/24 | 2 |

  The two gpt-oss models differ by one question (noise at n = 24); 120b was chosen as the larger
  model of the same family with the same limits. Groq answers take about 1-2 s per question in
  Docker vs 10-18 s for local Ollama. Module 4 numbers are Ollama numbers; if Groq results are
  reported, rerun the evaluation with Groq.
- **Groq free-plan limits** (from the API's rate-limit headers): **8,000 tokens per minute and
  1,000 requests per day, per model**. Prompts are about 200-1,350 tokens (schema included) and
  the gpt-oss models also spend tokens on hidden reasoning, so a burst of a few questions per
  minute hits HTTP 429; the API returns 429 with Retry-After and the UI shows the message.
  Enough for a demo, not for many simultaneous users.
- **Question data sent to Groq**: the prompt contains the schema and up to 3 sample values
  per column; the project's public datasets are fine, but user uploads (later) would leave
  the server. State this in the user guide.

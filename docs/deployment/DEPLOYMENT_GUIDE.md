# DataCopilot deployment guide

How to run DataCopilot locally in Docker and deploy it to AWS (Sydney, `ap-southeast-2`) with the
scripts in `infra/`. Design and costs: [ARCHITECTURE.md](ARCHITECTURE.md). End users:
[USER_GUIDE.md](USER_GUIDE.md).

> Never put secrets (API keys, passwords, tokens) in git, chat, issues or shared documents. They
> live only in your local `.env` (gitignored) and, once deployed, in SSM Parameter Store.

## 1. Prerequisites

| Tool | Version used | Notes |
|---|---|---|
| Git | any recent | On Windows, Git for Windows: its **Git Bash** runs the `infra/*.sh` scripts (or use WSL) |
| Python | 3.13 | For tests and local runs (`pip install -r requirements.txt`) |
| Docker Desktop | 29.x with Compose v5 and buildx | Images are built for `linux/amd64` |
| AWS CLI | v2 | Signed in to the project account (`aws login`), region `ap-southeast-2` |
| Session Manager plugin | 1.2.x | Only for a shell on the server (`aws ssm start-session`) |
| A DuckDNS account | free | One subdomain (e.g. `name.duckdns.org`) and its token |
| A Groq account | free | API key from console.groq.com |
| Processed data | `data/processed/` | Built by `scripts/prepare_data.py` (see `docs/notes/handoff.md`) or copied from the deployed bucket |

Account permissions needed: CloudFormation, EC2, IAM (role and instance profile), ECR, S3, SSM,
CloudWatch, Logs, SNS, Budgets in `ap-southeast-2`.

## 2. Environment setup

```bash
git clone https://github.com/mritunjaySharma05/datacopilot.git && cd datacopilot
python -m venv .venv && source .venv/bin/activate      # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
pytest                                                 # all tests run offline
aws login && aws configure set region ap-southeast-2
aws sts get-caller-identity                            # check you are in the project account
cp .env.example .env                                   # then fill it in (section 3)
```

On Windows, `.gitattributes` keeps the shell scripts and `infra/` files with LF line endings; do
not convert them to CRLF (the scripts would fail with `$'\r': command not found`).

### Data

`deploy.sh` uploads `data/processed/` to S3 and stops if it is missing. Either rebuild it
(`scripts/download_datasets.py`, `scripts/prepare_data.py`, then `scripts/build_questions.py --check`)
or copy the deployed copy back:

```bash
aws s3 sync s3://datacopilot-<account-id>-apse2/data/ data/processed/
```

## 3. `.env` keys

`.env` is read by the app (local runs), by `docker compose` and by `infra/deploy.sh`. Values are
never printed by the scripts. Names only:

| Key | Used by | Required | Meaning |
|---|---|---|---|
| `LLM_PROVIDER` | app | yes | `groq` (hosted, deployed) or `ollama` (local, Module 4 runs) |
| `GROQ_API_KEY` | app, deploy | for `groq` | Groq API key (secret) |
| `GROQ_MODEL` | app, deploy | for `groq` | Model id; the project uses `openai/gpt-oss-120b` |
| `OLLAMA_HOST` | app | for `ollama` | Ollama URL (Docker Compose overrides it with `host.docker.internal`) |
| `OLLAMA_MODEL` | app | for `ollama` | e.g. `qwen2.5-coder:7b` |
| `OLLAMA_NUM_CTX` | app | for `ollama` | Context window (8192) |
| `SITE_HOSTNAME` | deploy | yes | Public hostname, e.g. `name.duckdns.org` |
| `ALARM_EMAIL` | deploy | yes | Receives alarm and budget emails |
| `DUCKDNS_TOKEN` | deploy | recommended | Lets `deploy.sh` point the hostname at the server (secret; used only on your machine) |
| `BASIC_AUTH_USER` | deploy | no (default `demo`) | Site login name |
| `BASIC_AUTH_PASSWORD` | deploy | yes | Site login password, at least 12 characters (secret; only its bcrypt hash is uploaded) |
| `BUDGET_USD` | deploy | no (default 30) | Monthly AWS budget alert amount |
| `POSTGRES_URL` | (later) | no | Reserved for the PostgreSQL connection |
| `DATA_GOV_IN_API_KEY` | data scripts | no | Only for downloading crop data from data.gov.in |

Set by Docker Compose or the server, not in `.env`: `DATA_DIR` (data folder, `/data` in the
container), `MODEL_PATH` (classifier file), `API_URL` (UI to API), `LOG_LEVEL`, `GROQ_BASE_URL`
(optional override).

## 4. Run locally in Docker

```bash
docker compose up --build        # API :8000, UI http://localhost:8501
docker compose logs -f api       # one JSON line per question
docker compose down
```

With `LLM_PROVIDER=ollama`, Ollama runs on the host and the containers reach it at
`host.docker.internal:11434`.

## 5. AWS commands

Run from the repo root (bash / Git Bash). Region and stack name (`datacopilot`) are fixed in
`infra/lib.sh`.

| Command | What it does | Time |
|---|---|---|
| `infra/deploy.sh` | Creates or updates everything: validates and deploys the CloudFormation stack; writes the Groq key and login hash to SSM (SecureString, via a 0600 temp file); builds and pushes `linux/amd64` images tagged with the git commit; syncs `data/processed/` and the config to S3; points DuckDNS at the Elastic IP; restarts the app over SSM; waits until the site answers | first run ~10-15 min, updates ~5 min |
| `infra/status.sh` | Stack status, instance state, public IP, URL check (401 = up, login required), SSM agent, alarm state, remaining AWS credits | seconds |
| `infra/stop.sh` | Stops the instance (disk, IP and everything else kept; ~$0.17/day) | ~1 min |
| `infra/start.sh` | Starts it; the app starts itself and the script waits for the URL | 2-3 min |
| `infra/delete.sh` | Deletes everything (type `datacopilot` to confirm): empties the bucket, deletes the stack (instance, disk, IP, ECR images, logs, alarm, topic, budget, role, security group) and the two SSM parameters | 5-10 min |

Useful extras:

```bash
aws logs tail /datacopilot/api --follow                       # questions as JSON lines
aws logs tail /datacopilot/caddy --since 1h                   # certificates, HTTP access
aws ssm start-session --target <instance-id>                  # shell on the server, no SSH
#   on the server: sudo systemctl status datacopilot; sudo journalctl -u datacopilot -n 100
#                  sudo docker ps; sudo docker logs datacopilot-api-1
```

`deploy.sh` is idempotent: run it again after a code change (new image tag), a new Groq key or
password in `.env`, or a data change.

## 6. Validation after a deploy

| Check | How | Expected |
|---|---|---|
| Site is up and protected | `curl -s -o /dev/null -w '%{http_code}' https://<hostname>/` | `401` (login required) |
| HTTP redirects | `curl -s -o /dev/null -w '%{http_code}' http://<hostname>/` | `308` to `https://` |
| Certificate | Browser padlock, or `openssl s_client -connect <hostname>:443 -servername <hostname>` | Issuer Let's Encrypt, subject = your hostname |
| Login works | Open the URL, sign in | Streamlit page "DataCopilot" with the dataset picker |
| Only 80/443 open | `nc -z -G 4 <ip> 22`, `8000`, `8501` | All closed |
| Answers | Ask "Which state had the highest total rice production?" on **crop** | West Bengal, with the SQL |
| Refusal | Ask "Delete all rows for Kerala" | Refused by the gate, no SQL |
| Logs | `aws logs tail /datacopilot/api --since 10m` | One JSON line per question, with `gate_answerable`, `sql`, `latency_ms`; no secrets |
| Health of the stack | `infra/status.sh` | `CREATE_COMPLETE`/`UPDATE_COMPLETE`, instance `running`, SSM `Online`, alarm `OK` |
| Emails | Inbox of `ALARM_EMAIL` | Confirm the SNS subscription link once |

Pipeline accuracy of the deployed configuration (Groq) on the Module 4 test questions:
`docs/results/groq_vs_ollama.md` (51.1% strict vs 33.3% for Ollama, exact McNemar p = 0.0004).

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: set KEY in .env` | Missing `.env` value | Fill the key (section 3) |
| `data/processed/... missing` | No local data | Rebuild it or `aws s3 sync` it back (section 2) |
| `ExpiredToken` / `Unable to locate credentials` | AWS sign-in expired | `aws login` again |
| Stack `ROLLBACK_COMPLETE` after a failed first deploy | A resource failed to create | `aws cloudformation describe-stack-events --stack-name datacopilot`; fix the cause and run `deploy.sh` again (it deletes the rolled-back stack first) |
| `deploy.sh` waits and ends with "did not answer with the login prompt" | Hostname not pointing at the IP, certificate not issued yet, or the app still starting | `nslookup <hostname>` (should be the Elastic IP); `aws logs tail /datacopilot/caddy`; on the server `journalctl -u datacopilot` (the start script retries every 30 s) |
| `DuckDNS update failed` | Wrong `DUCKDNS_TOKEN` or subdomain | Check both on duckdns.org; or set the IP there by hand |
| Certificate errors after several delete/redeploy cycles | Let's Encrypt allows 5 certificates per hostname per week | Wait, or use `infra/stop.sh` / `start.sh` instead of delete/redeploy (the certificate is kept on the disk) |
| UI: "HTTP 429: Groq rate limit reached ... Try again in N s" | Groq free plan: 8,000 tokens/minute | Wait the given seconds and ask again |
| UI: "HTTP 503: Groq rejected the request ... check GROQ_API_KEY" (alarm email) | Key revoked or wrong | Put a new key in `.env`, run `deploy.sh` |
| UI: "Cannot reach the API" | `api` container down or restarting | `sudo docker ps` / `docker logs datacopilot-api-1` on the server; `sudo systemctl restart datacopilot` |
| Browser asks for the login again and again | Wrong user or password | Use the values from `.env`; after changing them run `deploy.sh` |
| `$'\r': command not found` (Windows) | Scripts checked out with CRLF | Commit or stash your changes first, then `git config core.autocrlf false` and `git rm --cached -r . && git reset --hard` (re-checks out every file; discards uncommitted changes) |
| `SessionManagerPlugin is not found` | Plugin not installed | Install the Session Manager plugin |
| No alarm emails | SNS subscription not confirmed | Click the confirmation link in the first AWS email |
| Bill higher than expected | Instance left running | `infra/status.sh` for credits; `infra/stop.sh` between demos; `infra/delete.sh` when finished |

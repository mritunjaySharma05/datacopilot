# Module 5 step 6: AWS deployment plan

Status: **deployed 2026-11-24** with `infra/deploy.sh` (approved choices: t3.small amd64; Caddy +
DuckDNS `datacopilot.duckdns.org` + basic auth; budget $30/month). Verified: Let's Encrypt
certificate, HTTP->HTTPS redirect, 401 without login, ports 22/8000/8501 closed, three questions
answered through the cloud API with Groq, JSON lines in CloudWatch with no secrets in any log group.
First attempt failed on an apostrophe in a security-group rule description (fixed in `0c45f57`).
Region **ap-southeast-2 (Sydney)**, account `<account-id>`, CLI profile `default`
(role `AccountFullAccessRole`).

## 0. Account facts (read-only checks, 2026-11-24)

| Check | Result |
|---|---|
| Plan | AWS Free plan, active, **$120.00 credits left**, plan expires **2027-05-23** (`aws freetier get-account-plan-state`) |
| App Runner | **Not usable**: an organization SCP explicitly denies `apprunner:*` in this account; AWS also closed App Runner to new customers from 2026-04-30 and points to ECS Express Mode instead |
| Other services needed | Read access works for EC2, ECR, S3, SSM, CloudWatch, Logs, IAM, SNS, Budgets, CloudFormation (create permissions are only proven by the first deploy) |
| Network | Default VPC `vpc-089054e6c859153f5` with public subnets in 2a/2b/2c |
| Instance types | `t3.small` (2 vCPU, 2 GB) and `t4g.small` are free-plan eligible (usage is paid from the credits) |
| AMI | Amazon Linux 2023 x86_64, resolved at deploy time from the public SSM parameter `/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64` |
| Local tools | AWS CLI 2.42.2; **Session Manager plugin missing** (`brew install --cask session-manager-plugin` before using a shell on the server) |

**Memory measured locally (Docker):** API 367 MB idle, **735 MB with all four datasets
loaded**; UI 60 MB (+ roughly 100 MB per active browser session). With Caddy, Docker and the OS,
about 1.3-1.5 GB, so `t3.micro` (1 GB) is too small; `t3.small` (2 GB) plus a 2 GB swap file.

## 1. Architecture

```
Internet --443/80--> [Elastic IP] EC2 t3.small (AL2023, Docker, no SSH)
                       |  caddy  (HTTPS + basic auth)  --> ui:8501 (Streamlit)  --> api:8000 (FastAPI)
                       |                                                              |-> Groq API (HTTPS out)
                       |  /opt/datacopilot/data  <-- S3 sync at start-up (read-only mount into api)
                       |  /run/datacopilot/*.env <-- SSM SecureString at start-up (tmpfs, never on disk)
                       |  container logs --> CloudWatch Logs (/datacopilot/api|ui|caddy)
```

- Only Caddy publishes ports (80, 443). The API and UI ports stay on the Docker network.
- A systemd unit `datacopilot.service` runs on every boot: fetch config and secrets, sync data,
  log in to ECR, `docker compose pull`, `docker compose up`. If images or data are not there
  yet, it retries every 30 s, so stop/start and first boot work without manual steps.
- Updates: `infra/deploy.sh` again (new image tag = git commit), which restarts the service via
  SSM Run Command.

## 2. Resource list

One CloudFormation stack, `datacopilot` (`infra/datacopilot.yaml`), plus two SSM parameters that
CloudFormation cannot create as SecureString (made by the deploy script).

| # | Resource | Settings |
|---|---|---|
| 1 | S3 bucket `datacopilot-<account>-apse2` | Block Public Access (all four), SSE-S3 encryption (free), bucket policy denying non-TLS access, no versioning. Prefixes `data/` (processed datasets, ~250 MB) and `config/` (compose file, Caddyfile, bootstrap script, image tag) |
| 2-3 | ECR repositories `datacopilot/api`, `datacopilot/ui` | Private, scan on push, lifecycle: keep last 3 images, `EmptyOnDelete` so the stack delete removes them |
| 4 | SSM parameter `/datacopilot/groq_api_key` | SecureString, AWS-managed key `aws/ssm`; value read from local `.env` by the script, passed via stdin/0600 temp file, never echoed or on the command line |
| 5 | SSM parameter `/datacopilot/basic_auth_hash` | SecureString; bcrypt hash only (the password itself is never stored) |
| 6 | IAM role + instance profile `datacopilot-instance` | Least privilege, see below |
| 7 | Security group `datacopilot-web` | Inbound TCP 80 and 443 from anywhere; **no 22**; outbound all (ECR, S3, Groq, Let's Encrypt, packages) |
| 8 | EC2 instance `t3.small` | AL2023, gp3 **16 GB encrypted** root volume, IMDSv2 required, no key pair, detailed monitoring off, termination protection off (so delete works) |
| 9 | Elastic IP | Stable address for the hostname |
| 10-12 | CloudWatch log groups `/datacopilot/api`, `/datacopilot/ui`, `/datacopilot/caddy` | Retention 14 days; Docker `awslogs` driver |
| 13 | Metric filter on `/datacopilot/api` | `{ $.http_status >= 500 }` -> metric `DataCopilot/Api5xxErrors` |
| 14 | CloudWatch alarm `datacopilot-api-5xx` | Sum >= 1 in 5 minutes (LLM unreachable, data missing, crashes) |
| 15 | SNS topic + email subscription | Alarm notifications (needs your email; you confirm the subscription once) |
| 16 | AWS Budgets monthly cost budget (recommended) | e.g. $30/month, email at 50% / 80% / 100% of actual cost. Free |

**Instance role permissions** (inline policy, no managed policies):
- `s3:ListBucket` on the bucket; `s3:GetObject` on `bucket/data/*` and `bucket/config/*`.
- `ssm:GetParameter` on `parameter/datacopilot/*` only (decryption with the AWS-managed `aws/ssm` key
  needs no extra KMS permission).
- `ecr:GetAuthorizationToken` (resource `*`, required by AWS); `ecr:BatchGetImage`,
  `ecr:GetDownloadUrlForLayer`, `ecr:BatchCheckLayerAvailability` on the two repositories.
- `logs:CreateLogStream`, `logs:PutLogEvents` on the three log groups.
- Session Manager, minimal: `ssm:UpdateInstanceInformation`, `ssmmessages:Create/OpenControlChannel`,
  `ssmmessages:Create/OpenDataChannel`, `ec2messages:*Message*`/`GetEndpoint`. The AWS managed
  policy `AmazonSSMManagedInstanceCore` is not used because it also allows `ssm:GetParameter` on
  every parameter in the account.
- Nothing else: no write to S3, no other secrets, no EC2/IAM actions.

## 3. Estimated cost (Sydney on-demand prices from the AWS Price List API, 2026-11-24)

| Item | Price | Per day | Per month (730 h) |
|---|---|---|---|
| EC2 `t3.small` | $0.0264 / h | $0.63 | $19.27 |
| Public IPv4 (Elastic IP) | $0.005 / h, in use or idle | $0.12 | $3.65 |
| EBS gp3 16 GB | $0.096 / GB-month | $0.05 | $1.54 |
| S3 ~0.3 GB | $0.025 / GB-month | ~$0.00 | $0.01 |
| ECR ~0.4 GB (compressed images) | $0.10 / GB-month | ~$0.00 | $0.04 |
| CloudWatch Logs (a few MB/month) + 1 alarm | ingestion per GB; $0.10 / alarm-month | ~$0.00 | ~$0.15 |
| SSM Parameter Store standard, `aws/ssm` key, SNS email, Budgets, data out (< 100 GB) | free | $0 | $0 |
| **Total, running 24/7** | | **~$0.81** | **~$24.7** |
| **Total, instance stopped** (EBS + Elastic IP still billed) | | **~$0.17** | **~$5.2** |

- $120 of credits covers about **148 days** of 24/7 running (to roughly mid-April 2027, before the
  plan's 2027-05-23 expiry). Stopping the instance between demos stretches this a lot.
- Running only for demos, e.g. 3 h/day: about $0.25/day.
- On the Free plan, AWS closes the account when the credits run out or the plan expires unless it is
  upgraded, so check credits with `infra/status.sh` and delete the stack when the module is done.
- Secrets Manager would add $0.40 per secret per month; Parameter Store standard is free and does
  the same job here, so Parameter Store is chosen.

**Options considered and not chosen:**

| Option | Why not |
|---|---|
| `t4g.small` (ARM, $0.0212/h, ~$15.5/month) | ~$3.8/month cheaper and builds natively on the Mac, but needs arm64 images instead of the amd64 ones already built. **Possible switch if you want it** (one parameter) |
| `t3.micro` (1 GB, ~$9.6/month) | Too small: the API alone needs ~735 MB with all datasets loaded |
| ECS Express Mode / Fargate | Needs a load balancer (~$18/month alone in Sydney) plus Fargate tasks; ~$40+/month |
| App Runner | Denied by the organization SCP; closed to new customers |
| Lightsail 2 GB (~$12/month) | Cheaper, but Lightsail instances cannot have an IAM role, so reading S3/SSM would need long-lived access keys on the server, and there is no Session Manager |

## 4. HTTPS and login (your decision)

**Recommended: Caddy + free DuckDNS hostname + basic auth.**

- **Hostname:** create a free subdomain at duckdns.org (sign in with GitHub/Google), e.g.
  `datacopilot-xyz.duckdns.org`, and point it at the Elastic IP once. `duckdns.org` is on the
  Public Suffix List (checked 2026-11-24), so Let's Encrypt's rate limits apply to your subdomain
  alone. Caddy obtains and renews the certificate automatically (HTTP-01 on port 80) and
  redirects HTTP to HTTPS.
- **Login:** Caddy `basic_auth` in front of the whole site, one user (e.g. `demo`) with a
  password you choose; only its bcrypt hash is stored (SSM SecureString). Over HTTPS this is
  acceptable for a demo, and it also stops strangers from using up the Groq quota (8k tokens/min).
- **Cost:** $0 extra. Caddy runs as a third container (official `caddy` 2.x image, exact tag
  pinned at implementation).

Alternatives:

| Option | Trade-off |
|---|---|
| `<ip>.sslip.io` / `nip.io` hostname (no sign-up) | Not on the Public Suffix List, so all users share Let's Encrypt's per-domain limit; issuance can fail with HTTP 429 |
| Let's Encrypt IP-address certificate | Supported by Let's Encrypt since 2025 but only as ~6-day certificates; Caddy support reported as unreliable. Not worth the risk for a demo |
| Own domain (Route 53 or other registrar) | Cleanest URL; ~$15/year for a domain plus $0.50/month for a Route 53 hosted zone |
| Streamlit's built-in OIDC login (`st.login`) or Cognito | Real user accounts, but needs an identity provider setup; Cognito with a load balancer adds ~$18/month. Could come later in Module 5/7 |
| HTTP only, no login | Not recommended: open to anyone, password-free Groq use |

## 5. Files to add (after approval)

```
infra/
  datacopilot.yaml        CloudFormation template (resources 1-3, 6-16)
  compose.aws.yml         api + ui + caddy, awslogs logging, read-only data mount, no published api/ui ports
  Caddyfile               {$SITE_HOSTNAME} { basic_auth ...; reverse_proxy ui:8501 }
  bootstrap.sh            runs on the instance before compose: config, secrets (to /run), data, ECR login
  deploy.sh  stop.sh  start.sh  status.sh  delete.sh
tests/test_infra_files.py static checks (no 22 open, IMDSv2, encryption, no secrets in files, least-privilege actions)
```

The template is validated with `aws cloudformation validate-template` (free) before any deploy.

## 6. Commands (all from the repo root; region and stack name fixed in the scripts)

**Deploy / update** (one command; prompts once for the basic-auth password):

```bash
infra/deploy.sh --hostname datacopilot-xyz.duckdns.org --alarm-email you@example.com
```

What it runs, in order:

```bash
aws cloudformation deploy --region ap-southeast-2 --stack-name datacopilot \
  --template-file infra/datacopilot.yaml --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides SiteHostname=... AlarmEmail=... InstanceType=t3.small
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.ap-southeast-2.amazonaws.com
docker buildx build --platform linux/amd64 -f docker/Dockerfile --target api -t <repo>/api:<git-sha> --push .
docker buildx build --platform linux/amd64 -f docker/Dockerfile --target ui  -t <repo>/ui:<git-sha>  --push .
aws s3 sync data/processed s3://<bucket>/data/ --delete
aws s3 cp infra/compose.aws.yml infra/Caddyfile infra/bootstrap.sh s3://<bucket>/config/   # + release.env with the image tag
aws ssm put-parameter --name /datacopilot/groq_api_key --type SecureString --overwrite --cli-input-json file://<0600 temp file>
aws ssm put-parameter --name /datacopilot/basic_auth_hash --type SecureString --overwrite --cli-input-json file://<0600 temp file>
aws ssm send-command --document-name AWS-RunShellScript --instance-ids <id> --parameters commands="systemctl restart datacopilot"
```

Then point the DuckDNS hostname at the Elastic IP printed by the script (first deploy only).

**Stop** (keeps everything; ~$0.17/day) and **start**:

```bash
infra/stop.sh     # aws ec2 stop-instances --instance-ids <id>  (+ wait)
infra/start.sh    # aws ec2 start-instances --instance-ids <id> (+ wait; the service starts itself)
infra/status.sh   # instance state, URL, last alarm state, remaining credits
```

**Full delete** (asks you to type `datacopilot` to confirm):

```bash
infra/delete.sh
#   aws s3 rm s3://<bucket> --recursive
#   aws cloudformation delete-stack --stack-name datacopilot && aws cloudformation wait stack-delete-complete ...
#   aws ssm delete-parameters --names /datacopilot/groq_api_key /datacopilot/basic_auth_hash
```

Deleting the stack removes the instance, its volume, the Elastic IP, ECR repositories (with images),
log groups, alarm, SNS topic, IAM role and security group. Nothing is left that bills.

**Shell on the server** (no SSH): `aws ssm start-session --target <instance-id>` (needs the
Session Manager plugin).

## 7. Decisions needed

1. Approve the plan (t3.small amd64), or switch to `t4g.small` (arm64, ~$3.8/month cheaper).
2. HTTPS/login: DuckDNS + Caddy + basic auth (recommended), or another option from section 4.
   If DuckDNS: create the subdomain and send me its name.
3. Email address for the alarm and budget notifications.
4. Budget alert amount (suggested $30/month).

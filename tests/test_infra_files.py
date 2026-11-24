"""Static checks on infra/ (no AWS calls): least privilege, no SSH, encryption, no secrets, scripts parse."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "infra"
TEMPLATE = (INFRA / "datacopilot.yaml").read_text()
COMPOSE = (INFRA / "compose.aws.yml").read_text()
SCRIPTS = sorted(INFRA.glob("*.sh"))
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")

ALLOWED_ACTIONS = {
    "sts:AssumeRole", "s3:ListBucket", "s3:GetObject", "ssm:GetParameter", "ecr:GetAuthorizationToken",
    "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability",
    "logs:CreateLogStream", "logs:PutLogEvents", "ssm:UpdateInstanceInformation",
    "ssmmessages:CreateControlChannel", "ssmmessages:CreateDataChannel", "ssmmessages:OpenControlChannel",
    "ssmmessages:OpenDataChannel", "ec2messages:AcknowledgeMessage", "ec2messages:DeleteMessage",
    "ec2messages:FailMessage", "ec2messages:GetEndpoint", "ec2messages:GetMessages", "ec2messages:SendReply",
}


# ---------- CloudFormation template ----------

def role_policy_actions() -> set[str]:
    role = TEMPLATE.split("  InstanceRole:")[1].split("  InstanceProfile:")[0]
    actions = set(re.findall(r"Action:\s*([a-z0-9]+:[A-Za-z*]+)", role))
    actions |= set(re.findall(r"^\s*-\s*([a-z0-9]+:[A-Za-z*]+)\s*$", role, flags=re.M))
    return actions


def test_instance_role_only_has_allowlisted_actions():
    actions = role_policy_actions()
    assert actions and actions <= ALLOWED_ACTIONS, actions - ALLOWED_ACTIONS
    assert not any(a.endswith(":*") for a in actions)
    assert "ManagedPolicyArns" not in TEMPLATE  # e.g. AmazonSSMManagedInstanceCore reads every parameter


def test_role_scopes_s3_and_ssm_to_own_resources():
    assert '"${DataBucket.Arn}/data/*"' in TEMPLATE and '"${DataBucket.Arn}/config/*"' in TEMPLATE
    assert "parameter/datacopilot/*" in TEMPLATE
    assert "s3:PutObject" not in TEMPLATE and "s3:DeleteObject" not in TEMPLATE


def test_only_http_and_https_open():
    ports = re.findall(r"FromPort:\s*(\d+)", TEMPLATE)
    assert sorted(ports) == ["443", "80"]
    assert "KeyName" not in TEMPLATE  # no SSH key pair


def test_instance_hardening_and_encryption():
    assert "HttpTokens: required" in TEMPLATE and "HttpPutResponseHopLimit: 1" in TEMPLATE
    assert "Encrypted: true" in TEMPLATE and "VolumeType: gp3" in TEMPLATE
    assert "AllowedValues: [t3.small]" in TEMPLATE and "al2023-ami-kernel-default-x86_64" in TEMPLATE
    for flag in ("BlockPublicAcls", "BlockPublicPolicy", "IgnorePublicAcls", "RestrictPublicBuckets"):
        assert f"{flag}: true" in TEMPLATE
    assert "SSEAlgorithm: AES256" in TEMPLATE and "aws:SecureTransport" in TEMPLATE


def test_ecr_logs_alarm_budget():
    assert TEMPLATE.count("EmptyOnDelete: true") == 2 and TEMPLATE.count("ScanOnPush: true") == 2
    assert TEMPLATE.count("RetentionInDays: 14") == 3
    assert '"{ $.http_status >= 500 }"' in TEMPLATE and "AlarmActions: [!Ref AlarmTopic]" in TEMPLATE
    assert "AWS::Budgets::Budget" in TEMPLATE and "Default: 30" in TEMPLATE


def test_user_data_pins_and_verifies_compose():
    assert "docker/compose/releases/download/v5.7.0/docker-compose-linux-x86_64" in TEMPLATE
    assert "sha256sum -c -" in TEMPLATE
    # tracing would print secrets
    assert not re.search(r"^\s*set\s+-\w*x", (INFRA / "bootstrap.sh").read_text(), flags=re.M)


# ---------- compose file and Caddyfile ----------

def test_compose_publishes_only_caddy_and_reads_secrets_from_tmpfs():
    services = re.split(r"^  (?=\w+:\n)", COMPOSE.split("\nservices:\n")[1].split("\nvolumes:\n")[0], flags=re.M)
    by_name = {s.split(":", 1)[0]: s for s in services if s.strip()}
    assert set(by_name) == {"api", "ui", "caddy"}
    assert "ports:" in by_name["caddy"] and "ports:" not in by_name["api"] and "ports:" not in by_name["ui"]
    assert "/run/datacopilot/api.env" in by_name["api"] and "/opt/datacopilot/data:/data:ro" in by_name["api"]
    assert "caddy:2.11.9" in by_name["caddy"]
    assert COMPOSE.count("awslogs-group: /datacopilot/") == 3


def test_caddyfile_template_has_auth_and_proxy():
    caddy = (INFRA / "Caddyfile").read_text()
    for token in ("__SITE_HOSTNAME__", "__AUTH_USER__ __AUTH_HASH__", "basic_auth", "reverse_proxy ui:8501"):
        assert token in caddy


# ---------- scripts ----------

def test_no_secret_values_or_echoes_in_infra():
    for path in INFRA.iterdir():
        if path.is_file():
            text = path.read_text()
            assert not re.search(r"gsk_[A-Za-z0-9]{8,}|\$2[aby]\$\d\d\$", text), path.name
            for line in text.splitlines():  # a secret may only be printed into a file, never to the terminal
                if re.search(r"\b(echo|printf)\b.*\$\{?(GROQ_API_KEY|BASIC_AUTH_PASSWORD|DUCKDNS_TOKEN)\b", line):
                    assert re.search(r'>\s*"\$RUN/', line), f"{path.name}: {line.strip()}"


def test_secret_temp_dir_is_gitignored():
    assert "infra/.secrets-tmp/" in (ROOT / ".gitignore").read_text()


@needs_bash
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_scripts_parse(script):
    assert subprocess.run([BASH, "-n", str(script)], capture_output=True).returncode == 0


@needs_bash
def test_env_get_parses_dotenv_like_python_dotenv(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=plain\nB = spaced   # comment\nC=\"quoted # not comment\"\nD='single'\n"
                   "export E=exported\nF=has#hash\nG=\nA=last\r\nH=crlf\r\n")
    script = "source infra/lib.sh; for k in A B C D E F G H MISSING; do printf '%s|' \"$(env_get $k dflt)\"; done"
    out = subprocess.run([BASH, "-c", script], cwd=ROOT, capture_output=True, text=True,
                         env={"ENV_FILE": str(env), "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.split("|")[:-1] == ["last", "spaced", "quoted # not comment", "single", "exported",
                                          "has#hash", "dflt", "crlf", "dflt"]


@needs_bash
def test_require_env_names_the_missing_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=x\n")
    out = subprocess.run([BASH, "-c", "source infra/lib.sh; require_env GROQ_API_KEY SITE_HOSTNAME"], cwd=ROOT,
                         capture_output=True, text=True, env={"ENV_FILE": str(env), "PATH": "/usr/bin:/bin"})
    assert out.returncode != 0 and "set SITE_HOSTNAME" in out.stderr


def test_gitattributes_forces_lf_for_scripts_and_infra():
    attrs = (ROOT / ".gitattributes").read_text()
    for pattern in ("*.sh", "infra/**", "docker/**"):
        assert re.search(rf"^{re.escape(pattern)}\s+text eol=lf", attrs, flags=re.M), pattern
    for path in [*SCRIPTS, INFRA / "Caddyfile", INFRA / "compose.aws.yml"]:
        assert b"\r\n" not in path.read_bytes(), path.name

"""Reuse a previously accepted AMLT config; freeze allowlisted small inputs."""
from pathlib import Path
import hashlib
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "amlt/feeds_c4_freegen_novel_seed42_20260905.yaml"
DEST = ROOT / "amlt/feeds_c4_format_diagnostic_seed42_20260905.yaml"
FILES = [
    "requirements-feeds.txt", "scripts/copy_model_parallel.py",
    "scripts/state_swap_misconception_smoke.py", "scripts/c4_freegen_protocol.py",
    "scripts/evaluate_c4_freegen_novel.py", "scripts/c4_format_diagnostic.py",
    "scripts/evaluate_c4_format_diagnostic.py", "scripts/run_c4_format_diagnostic.sh",
    "tests/test_c4_freegen_decode.py", "tests/test_c4_format_diagnostic.py",
    "manifests/c4_freegen_novel_seed42_20260905.json",
    "manifests/c4_format_diagnostic_seed42_20260905.json",
]
HASHFILE = "manifests/c4_format_diagnostic_20260905.sha256"


def main():
    config = yaml.safe_load(SOURCE.read_text())
    assert config["target"]["name"] == "Feeds"
    assert config["environment"]["image"] == "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
    assert all(j["sku"] == "80G1-A100" and j["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] == "rankfun_aml" for j in config["jobs"])
    config["description"] = "C4 frozen 81-query factorial: candidate and correct-answer support x scoring and free generation; no training"
    ignore = ["*", "!requirements-feeds.txt"]
    for directory in ("scripts", "tests", "manifests"):
        ignore += [f"!{directory}/", f"{directory}/*"]
        ignore += [f"!{p}" for p in FILES + [HASHFILE] if p.startswith(directory + "/")]
    ignore += ["__pycache__/", "*.pyc"]
    config["code"]["ignore"] = ignore
    for i, job in enumerate(config["jobs"]):
        job["name"] = f"c4-format-diagnostic-seed42-shard{i}"
        job["command"] = f"bash scripts/run_c4_format_diagnostic.sh {i}"
    text = "# Derived from accepted feeds_c4_freegen_novel_seed42_20260905.yaml.\n" + yaml.safe_dump(config, sort_keys=False)
    assert "REQUIRED" not in text and "latest" not in text
    DEST.write_text(text)
    hashes = "".join(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() + "  " + p + "\n" for p in FILES)
    (ROOT / HASHFILE).write_text(hashes)
    print(DEST)
    print(f"Frozen {len(FILES)} input/code files; upload allowlist includes hash manifest.")


if __name__ == "__main__":
    main()

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_profile_transport_amlt_is_bound_to_frozen_inputs():
    path = ROOT / "amlt/feeds_dbe_student_profile_transport_20260816.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["target"]["name"] == "Feeds"
    assert "CS-NewsAndFeeds-Singularity" in config["target"]["workspace_name"]
    assert config["environment"]["image"] == (
        "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
    )
    assert len(config["jobs"]) == 1
    job = config["jobs"][0]
    assert (job["sku"], job["process_count_per_node"], job["mpi"], job["sla_tier"]) == (
        "80G1-A100",
        1,
        False,
        "Standard",
    )
    assert job["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] == "rankfun_aml"
    runner = ROOT / "scripts/run_dbe_student_profile_transport.sh"
    digest = hashlib.sha256(runner.read_bytes()).hexdigest()
    command = "\n".join(job["command"] if isinstance(job["command"], list) else [job["command"]])
    assert digest in command

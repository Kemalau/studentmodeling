"""Read-only remote monitor, download, and complete-panel summary; never submits."""
from pathlib import Path
import json
import os
import subprocess
import time

from c4_freegen_protocol import summarize

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "c4-freegen-novel-seed42-v1-20260905"
PREFIX = ROOT / "results/c4_freegen_novel_seed42_20260905"


def command(argv, timeout=90):
    return subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def write(suffix, payload):
    path = Path(str(PREFIX) + suffix)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def main():
    write(".monitor.json", {"pid": os.getpid(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "experiment": EXPERIMENT, "status": "monitoring"})
    while True:
        try:
            result = command(["amlt", "--json-tables", "status", EXPERIMENT, "--force-status-update"])
            if result.returncode:
                raise RuntimeError(result.stderr + result.stdout)
            jobs = json.loads(result.stdout[result.stdout.index("["):])
            write(".status.json", {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "jobs": jobs})
            states = [str(j["status"]).lower() for j in jobs]
            print(time.strftime("%H:%M:%S", time.gmtime()), states, flush=True)
            if len(jobs) != 3:
                raise ValueError("Expected exactly three jobs")
            if all(s == "pass" for s in states):
                result = command(["amlt", "results", "download", EXPERIMENT, "--output", str(ROOT / "results/downloads")], timeout=180)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                files = sorted((ROOT / "results/downloads" / EXPERIMENT).rglob("freegen/records.jsonl"))
                if len(files) != 3:
                    raise ValueError(f"Expected three result shards, found {len(files)}")
                manifest = json.loads((ROOT / "manifests/c4_freegen_novel_seed42_20260905.json").read_text())
                records = [json.loads(line) for path in files for line in path.open() if line.strip()]
                summary = summarize(manifest, records)
                summary["files"] = [str(p) for p in files]
                summary["experiment"] = EXPERIMENT
                write(".summary.json", summary)
                write(".monitor.json", {"pid": os.getpid(), "status": "complete", "experiment": EXPERIMENT})
                print(json.dumps(summary, indent=2), flush=True)
                return
            if any(s in {"failed", "killed", "cancelled", "canceled"} for s in states):
                write(".monitor.json", {"pid": os.getpid(), "status": "needs_attention", "jobs": jobs})
                return
        except Exception as exc:
            print(type(exc).__name__, str(exc)[:1800], flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()

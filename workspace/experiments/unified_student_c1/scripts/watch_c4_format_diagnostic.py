"""Native AMLT terminal-state watch, then download and audit; never submits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from c4_format_diagnostic import collect

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "c4-format-diagnostic-seed42-v1-20260905"
PREFIX = ROOT / "results/c4_format_diagnostic_seed42_20260905"


def write(suffix, payload):
    p = Path(str(PREFIX) + suffix)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(t, p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    if args.background:
        receipt = Path(str(PREFIX) + ".monitor.json")
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous.get("status") == "monitoring":
                try:
                    os.kill(previous["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError("Monitor already alive")
        with Path(str(PREFIX) + ".monitor.log").open("a") as log:
            proc = subprocess.Popen([sys.executable, "-u", __file__], cwd=ROOT,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        print(json.dumps({"monitor_pid": proc.pid, "experiment": EXPERIMENT}))
        return
    write(".monitor.json", {"pid": os.getpid(), "status": "monitoring", "experiment": EXPERIMENT,
                            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    try:
        result = subprocess.run(["amlt", "watch", EXPERIMENT, "--json", "--interval", "60",
                                 "--notify", "terminal", "--state-file", str(PREFIX) + ".watch.json"], cwd=ROOT)
        if result.returncode:
            raise RuntimeError(f"amlt watch exit={result.returncode}; inspect monitor log")
        downloaded = subprocess.run(["amlt", "results", "download", EXPERIMENT, "--output", str(ROOT / "results/downloads")],
                                    cwd=ROOT, timeout=300)
        if downloaded.returncode:
            raise RuntimeError(f"download exit={downloaded.returncode}")
        summary = collect(ROOT / "manifests/c4_format_diagnostic_seed42_20260905.json",
                          ROOT / "results/downloads" / EXPERIMENT)
        summary["experiment"] = EXPERIMENT
        write(".summary.json", summary)
        write(".monitor.json", {"pid": os.getpid(), "status": "complete", "experiment": EXPERIMENT})
        print(json.dumps({"status": "complete", "summary": str(PREFIX) + ".summary.json"}), flush=True)
    except Exception as exc:
        write(".monitor.json", {"pid": os.getpid(), "status": "needs_attention", "experiment": EXPERIMENT, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()

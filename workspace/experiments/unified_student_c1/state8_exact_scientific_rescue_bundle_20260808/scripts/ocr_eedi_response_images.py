#!/usr/bin/env python3
"""OCR Eedi response question images into text and option fields.

This script is intentionally separate from the C1 data builder: it creates a
question-level OCR manifest that can be reused by multiple dataset variants.
It expects RapidOCR to be installed in the Python environment used to run it.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm


LETTERS = ["A", "B", "C", "D"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def metadata_questions(raw_dir: Path) -> dict[str, dict[str, Any]]:
    by_question: dict[str, dict[str, Any]] = {}
    for metadata_path in sorted(raw_dir.glob("*/metadata.jsonl")):
        family_dir = metadata_path.parent
        for row in read_jsonl(metadata_path):
            qid = str(row["QuestionId"])
            if qid in by_question:
                continue
            image_name = str(row.get("img_file_name") or f"{qid}.jpg")
            image_path = family_dir / image_name
            by_question[qid] = {
                "question_id": qid,
                "macro_area": row.get("MacroArea") or family_dir.name.replace("_", " "),
                "subjects": row.get("Subjects") or [],
                "image_path": str(image_path),
                "image_exists": image_path.exists(),
            }
    return by_question


def box_stats(box: list[list[float]]) -> dict[str, float]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "x_mid": (min(xs) + max(xs)) / 2.0,
        "y_mid": (min(ys) + max(ys)) / 2.0,
    }


def clean_token(text: str) -> str:
    text = text.strip()
    text = text.replace("I=Eedi", "Eedi")
    text = re.sub(r"\s+", " ", text)
    return text


def should_drop_header_footer(text: str, y_mid: float, height: int) -> bool:
    packed = re.sub(r"[^a-z0-9]", "", text.lower())
    if "eedi" in packed and (y_mid < 0.12 * height or y_mid > 0.90 * height):
        return True
    if packed in {"", "eedi", "eedi2018"}:
        return True
    return False


def option_letter_for_x(x_mid: float, width: int) -> str:
    idx = min(3, max(0, int(x_mid / max(width, 1) * 4)))
    return LETTERS[idx]


def infer_option_y_min(items: list[dict[str, Any]], height: int) -> float:
    if len(items) < 2:
        return 0.62 * height
    ys = sorted(float(item["y_mid"]) for item in items)
    for i in range(len(ys) - 1):
        gap = ys[i + 1] - ys[i]
        if ys[i + 1] > 0.42 * height and gap > 0.10 * height:
            return (ys[i] + ys[i + 1]) / 2.0
    candidates = []
    for i in range(len(ys) - 1):
        gap = ys[i + 1] - ys[i]
        if ys[i + 1] > 0.25 * height and gap > 0.055 * height:
            candidates.append((gap, i))
    if not candidates:
        return 0.62 * height
    _, idx = max(candidates)
    return (ys[idx] + ys[idx + 1]) / 2.0


def option_layout_is_grid(items: list[dict[str, Any]], height: int) -> bool:
    if len(items) < 3:
        return False
    ys = [float(item["y_mid"]) for item in items]
    return (max(ys) - min(ys)) > 0.18 * height


def assign_option_letter(item: dict[str, Any], items: list[dict[str, Any]], width: int, height: int) -> str:
    if not option_layout_is_grid(items, height):
        return option_letter_for_x(item["x_mid"], width)
    ys = sorted(float(x["y_mid"]) for x in items)
    mid = len(ys) // 2
    y_median = (ys[mid - 1] + ys[mid]) / 2.0 if len(ys) % 2 == 0 else ys[mid]
    top = item["y_mid"] <= y_median
    left = item["x_mid"] < 0.5 * width
    if top and left:
        return "A"
    if top and not left:
        return "B"
    if not top and left:
        return "C"
    return "D"


def join_items(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    items = sorted(items, key=lambda x: (round(x["y_mid"] / 12), x["x_mid"]))
    parts = [str(item["text"]).strip() for item in items if str(item["text"]).strip()]
    return " ".join(parts).strip()


def parse_ocr(raw_result: list[Any] | None, width: int, height: int) -> dict[str, Any]:
    raw_lines: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for item in raw_result or []:
        box, text, score = item
        stats = box_stats(box)
        text = clean_token(str(text))
        entry = {
            "box": box,
            "text": text,
            "score": float(score),
            **stats,
        }
        raw_lines.append(entry)
        if should_drop_header_footer(text, stats["y_mid"], height):
            continue
        if text in LETTERS and stats["y_mid"] > 0.45 * height:
            continue
        kept.append(entry)

    content_items = [x for x in kept if not (x["text"] in LETTERS and x["y_mid"] > 0.45 * height)]
    option_y_min = infer_option_y_min(content_items, height)
    option_items = [x for x in kept if x["y_mid"] >= option_y_min]
    question_items = [x for x in kept if x["y_mid"] < option_y_min]

    grouped: dict[str, list[dict[str, Any]]] = {letter: [] for letter in LETTERS}
    for item in option_items:
        grouped[assign_option_letter(item, option_items, width, height)].append(item)

    options = {letter: join_items(grouped[letter]) for letter in LETTERS}
    nonempty_options = sum(1 for text in options.values() if text)
    if nonempty_options < 2:
        question_items = content_items
        options = {letter: "" for letter in LETTERS}
        nonempty_options = 0
    question_text = join_items(question_items)
    raw_text = "\n".join(item["text"] for item in sorted(kept, key=lambda x: (x["y_mid"], x["x_mid"])))
    scores = [item["score"] for item in kept]
    return {
        "question_text_ocr": question_text,
        "options_ocr": options,
        "raw_text_ocr": raw_text,
        "raw_lines": raw_lines,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "n_lines": len(kept),
        "n_options_nonempty": nonempty_options,
        "parse_status": "ok" if question_text and nonempty_options >= 3 else "weak",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/unified_student_c1/data/raw/eedi_response_hf"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("experiments/unified_student_c1/data/processed/eedi_response_ocr/questions_ocr.jsonl"))
    parser.add_argument("--stats-json", type=Path, default=Path("experiments/unified_student_c1/data/processed/eedi_response_ocr/ocr_stats.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    questions = metadata_questions(args.raw_dir)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if args.resume and args.out_jsonl.exists():
        rows = read_jsonl(args.out_jsonl)
        seen = {str(row["question_id"]) for row in rows}

    todo = [q for qid, q in sorted(questions.items(), key=lambda kv: int(kv[0])) if qid not in seen]
    if args.limit:
        todo = todo[: args.limit]

    ocr = RapidOCR()
    for q in tqdm(todo, desc="OCR Eedi images"):
        image_path = Path(q["image_path"])
        row = dict(q)
        if not image_path.exists():
            row.update(
                {
                    "question_text_ocr": "",
                    "options_ocr": {letter: "" for letter in LETTERS},
                    "raw_text_ocr": "",
                    "raw_lines": [],
                    "avg_score": 0.0,
                    "n_lines": 0,
                    "n_options_nonempty": 0,
                    "parse_status": "missing_image",
                }
            )
            rows.append(row)
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        result, elapsed = ocr(str(image_path))
        row.update({"image_width": width, "image_height": height, "ocr_elapsed": elapsed})
        row.update(parse_ocr(result, width, height))
        rows.append(row)

    rows.sort(key=lambda r: int(r["question_id"]))
    write_jsonl(args.out_jsonl, rows)

    status_counts = defaultdict(int)
    option_counts = defaultdict(int)
    for row in rows:
        status_counts[row["parse_status"]] += 1
        option_counts[str(row["n_options_nonempty"])] += 1
    stats = {
        "n_questions": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "option_nonempty_counts": dict(sorted(option_counts.items())),
        "avg_ocr_score": sum(float(r.get("avg_score", 0.0)) for r in rows) / len(rows) if rows else 0.0,
        "raw_dir": str(args.raw_dir),
        "out_jsonl": str(args.out_jsonl),
    }
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

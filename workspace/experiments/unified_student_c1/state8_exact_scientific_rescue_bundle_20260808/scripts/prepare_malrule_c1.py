#!/usr/bin/env python3
"""Prepare cleaned MalruleLib C1 samples.

This script builds synthetic student histories from executable malrules and
applies the three data-cleaning filters needed for rule-level experiments:

1. manifestation: malrule answer differs from the correct answer;
2. answer consistency: metadata expected/correct answer agrees with the
   surface module's correct algorithm when such metadata exists;
3. domain consistency: when a malrule is applied to another generator's
   problem, both correct algorithms must agree.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FAMILIES_DEFAULT = ["subtraction", "decimals", "fractions"]
LEVELS_DEFAULT = [9, 8, 7, 6, 5, 4, 3, 2, 1]
DIFFICULTY_DEFAULT = "MEDIUM"
TASKS = ["malrule_next_response", "malrule_identity", "malrule_next_correctness"]
STATE_TOKEN = "<STUDENT_STATE>"


@dataclass(frozen=True)
class MalruleInfo:
    family: str
    name: str
    malrule_id: str
    module_name: str
    generator_cls: type
    malrule_cls: type
    correct_cls: type


@dataclass(frozen=True)
class Surface:
    family: str
    owner: str
    problem_type: str
    generator_cls: type
    correct_cls: type

    @property
    def surface_id(self) -> str:
        return f"{self.family}.{self.owner}.{self.problem_type}"


def snake_to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def norm_answer(value: Any) -> str:
    text = str(value).strip()
    text = " ".join(text.split())
    return text


def class_candidates(module: Any, base: type) -> list[tuple[str, type]]:
    out: list[tuple[str, type]] = []
    seen: set[type] = set()
    for name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        try:
            ok = issubclass(obj, base) and obj is not base
        except TypeError:
            ok = False
        if ok and obj not in seen:
            out.append((name, obj))
            seen.add(obj)
    return out


def pick_class(module: Any, base: type, directory_name: str, kind: str) -> tuple[str, type]:
    candidates = class_candidates(module, base)
    if not candidates:
        raise ValueError(f"No {kind} class found in {module.__name__}")

    camel = snake_to_camel(directory_name)
    preferred_names = []
    if kind == "generator":
        preferred_names = [f"{camel}ProblemGenerator"]
    elif kind == "malrule":
        preferred_names = [f"{camel}Malrule"]
    elif kind == "correct":
        preferred_names = [f"{camel}CorrectAlgorithm"]

    for wanted in preferred_names:
        for name, cls in candidates:
            if name == wanted:
                return name, cls

    if kind == "correct":
        for name, cls in candidates:
            if "Correct" in name and "Algorithm" in name:
                return name, cls
    if kind == "generator":
        for name, cls in candidates:
            if name.endswith("ProblemGenerator"):
                return name, cls
    if kind == "malrule":
        for name, cls in candidates:
            if name.endswith("Malrule"):
                return name, cls
    return candidates[0]


def discover_malrules(repo: Path, families: list[str]) -> list[MalruleInfo]:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils import CorrectAlgorithm, Malrule, ProblemGenerator

    infos: list[MalruleInfo] = []
    for family in families:
        fam_dir = repo / "malrules" / family
        if not fam_dir.exists():
            continue
        for mal_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir() and (p / "__init__.py").exists()):
            module_name = f"malrules.{family}.{mal_dir.name}"
            module = importlib.import_module(module_name)
            _, generator_cls = pick_class(module, ProblemGenerator, mal_dir.name, "generator")
            _, malrule_cls = pick_class(module, Malrule, mal_dir.name, "malrule")
            _, correct_cls = pick_class(module, CorrectAlgorithm, mal_dir.name, "correct")
            infos.append(
                MalruleInfo(
                    family=family,
                    name=mal_dir.name,
                    malrule_id=f"{family}.{mal_dir.name}",
                    module_name=module_name,
                    generator_cls=generator_cls,
                    malrule_cls=malrule_cls,
                    correct_cls=correct_cls,
                )
            )
    return infos


def difficulty_enum(repo: Path, name: str) -> Any:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils import DifficultyLevel

    return getattr(DifficultyLevel, name.upper())


def generate_problem(
    generator_cls: type,
    seed: int,
    difficulty: Any,
    levels: list[int],
) -> tuple[Any, int]:
    last_error: Exception | None = None
    for level in levels:
        try:
            generator = generator_cls(level=level, difficulty=difficulty)
            problem = generator.generate(seed)
            return problem, level
        except Exception as exc:  # noqa: BLE001 - malrulelib modules raise many ad hoc errors.
            last_error = exc
    raise RuntimeError(f"Could not generate with {generator_cls}: {last_error}")


def surface_type(problem: Any) -> str:
    metadata = problem.metadata or {}
    raw = (
        metadata.get("problem_type")
        or metadata.get("template")
        or metadata.get("formula_type")
        or metadata.get("operation")
        or "default"
    )
    return str(raw)


def solve_answer(algo_cls: type, problem: Any) -> str:
    return norm_answer(algo_cls().solve(problem).answer)


def metadata_expected_answer(problem: Any) -> str | None:
    metadata = problem.metadata or {}
    for key in ("expected_answer", "correct_answer"):
        if key in metadata and metadata[key] is not None:
            return norm_answer(metadata[key])
    return None


def solve_malrule_answer(malrule_cls: type, problem: Any) -> str:
    return norm_answer(malrule_cls().solve(problem).answer)


def is_consistent_with_metadata(problem: Any, correct_answer: str) -> bool:
    expected = metadata_expected_answer(problem)
    return expected is None or expected == correct_answer


def build_surfaces(
    malrules: list[MalruleInfo],
    difficulty: Any,
    levels: list[int],
    probe_seeds: int,
    rng: random.Random,
) -> tuple[list[Surface], dict[str, Any]]:
    counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    class_lookup: dict[tuple[str, str], tuple[type, type]] = {}
    errors: Counter[str] = Counter()

    for info in malrules:
        class_lookup[(info.family, info.name)] = (info.generator_cls, info.correct_cls)
        for _ in range(probe_seeds):
            seed = rng.randrange(1_000_000_000)
            try:
                problem, _ = generate_problem(info.generator_cls, seed, difficulty, levels)
                counts[(info.family, info.name, surface_type(problem))]["count"] += 1
            except Exception as exc:  # noqa: BLE001
                errors[f"{info.malrule_id}:{type(exc).__name__}"] += 1

    surfaces: list[Surface] = []
    for (family, owner, problem_type), counter in sorted(counts.items()):
        if counter["count"] < 4:
            continue
        generator_cls, correct_cls = class_lookup[(family, owner)]
        surfaces.append(
            Surface(
                family=family,
                owner=owner,
                problem_type=problem_type,
                generator_cls=generator_cls,
                correct_cls=correct_cls,
            )
        )

    stats = {
        "surface_probe_seeds_per_generator": probe_seeds,
        "surface_count_before_low_frequency_filter": len(counts),
        "surface_count_after_low_frequency_filter": len(surfaces),
        "surface_generation_errors": dict(errors.most_common(20)),
    }
    return surfaces, stats


def split_surfaces(surfaces: list[Surface]) -> tuple[list[Surface], list[Surface]]:
    side_a: list[Surface] = []
    side_b: list[Surface] = []
    by_family: dict[str, list[Surface]] = defaultdict(list)
    for surface in surfaces:
        by_family[surface.family].append(surface)
    for family, fam_surfaces in sorted(by_family.items()):
        ordered = sorted(fam_surfaces, key=lambda s: s.surface_id)
        for idx, surface in enumerate(ordered):
            if (idx + 1) % 3 == 0:
                side_b.append(surface)
            else:
                side_a.append(surface)
        if not any(s.family == family for s in side_b) and ordered:
            moved = side_a.pop()
            side_b.append(moved)
    return side_a, side_b


def try_clean_problem(
    malrule: MalruleInfo,
    surface: Surface,
    seed: int,
    difficulty: Any,
    levels: list[int],
) -> dict[str, Any] | None:
    try:
        problem, level = generate_problem(surface.generator_cls, seed, difficulty, levels)
        surface_correct = solve_answer(surface.correct_cls, problem)
        if not is_consistent_with_metadata(problem, surface_correct):
            return None
        if surface.owner != malrule.name:
            malrule_correct = solve_answer(malrule.correct_cls, problem)
            if malrule_correct != surface_correct:
                return None
        malrule_answer = solve_malrule_answer(malrule.malrule_cls, problem)
        if malrule_answer == surface_correct:
            return None
        return {
            "problem": problem,
            "level": level,
            "correct_answer": surface_correct,
            "malrule_answer": malrule_answer,
        }
    except Exception:  # noqa: BLE001
        return None


def sample_clean_problem(
    malrule: MalruleInfo,
    surface: Surface,
    rng: random.Random,
    difficulty: Any,
    levels: list[int],
    max_attempts: int,
) -> dict[str, Any] | None:
    for _ in range(max_attempts):
        seed = rng.randrange(1_000_000_000)
        item = try_clean_problem(malrule, surface, seed, difficulty, levels)
        if item is not None:
            item["seed"] = seed
            return item
    return None


def find_allowed_surfaces(
    malrules: list[MalruleInfo],
    surfaces: list[Surface],
    difficulty: Any,
    levels: list[int],
    max_attempts: int,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    rng = random.Random(seed)
    by_id = {surface.surface_id: surface for surface in surfaces}
    allowed: dict[str, list[str]] = {}
    attempts = 0
    for malrule in malrules:
        ok: list[str] = []
        for surface in surfaces:
            if surface.family != malrule.family:
                continue
            found = False
            for _ in range(max_attempts):
                attempts += 1
                item = try_clean_problem(
                    malrule,
                    surface,
                    rng.randrange(1_000_000_000),
                    difficulty,
                    levels,
                )
                if item is not None:
                    found = True
                    break
            if found:
                ok.append(surface.surface_id)
        allowed[malrule.malrule_id] = sorted(ok)
    stats = {
        "compatibility_attempts": attempts,
        "allowed_surface_counts": {k: len(v) for k, v in sorted(allowed.items())},
        "surfaces_indexed": len(by_id),
    }
    return allowed, stats


def problem_record(item: dict[str, Any], surface: Surface, student_answer: str, is_correct: bool) -> dict[str, Any]:
    problem = item["problem"]
    return {
        "problem_text": str(problem.text),
        "student_answer": student_answer,
        "correct_answer": item["correct_answer"],
        "is_correct": bool(is_correct),
        "surface_id": surface.surface_id,
        "surface_owner": surface.owner,
        "problem_type": surface.problem_type,
        "operation": str(getattr(problem, "operation", "")),
        "operands": [str(x) for x in getattr(problem, "operands", [])],
        "metadata": {str(k): str(v) for k, v in (problem.metadata or {}).items()},
        "generation_seed": item.get("seed"),
        "level": item.get("level"),
    }


def answers_by_family_malrule(
    query_item: dict[str, Any],
    surface: Surface,
    malrules: list[MalruleInfo],
) -> dict[str, str]:
    problem = query_item["problem"]
    out: dict[str, str] = {}
    surface_correct = query_item["correct_answer"]
    for info in malrules:
        if info.family != surface.family:
            continue
        try:
            if surface.owner != info.name:
                info_correct = solve_answer(info.correct_cls, problem)
                if info_correct != surface_correct:
                    continue
            out[info.malrule_id] = solve_malrule_answer(info.malrule_cls, problem)
        except Exception:  # noqa: BLE001
            continue
    return out


def build_segments(
    task: str,
    history: list[dict[str, Any]],
    query: dict[str, Any],
    candidate_targets: dict[str, str],
) -> tuple[str, str, str, str]:
    if task == "malrule_next_response":
        instruction = (
            "You are modeling one student's mathematical behavior. "
            "Predict the exact answer this same student would give on the next problem. "
            "Return exactly one <predicted_response> tag."
        )
    elif task == "malrule_identity":
        instruction = (
            "You are modeling one student's mathematical behavior. "
            "Diagnose which systematic misconception best explains the student's history. "
            "Return exactly one <misconception> tag."
        )
    elif task == "malrule_next_correctness":
        instruction = (
            "You are modeling one student's mathematical behavior. "
            "Predict whether the student will answer the next problem correctly. "
            "Return exactly one <correctness> tag."
        )
    else:
        raise ValueError(task)

    history_lines = []
    for idx, event in enumerate(history, 1):
        history_lines.extend(
            [
                f"Trial {idx}:",
                f"Problem: {event['problem_text']}",
                f"Student answer: {event['student_answer']}",
                f"Correct answer: {event['correct_answer']}",
                f"Outcome: {'correct' if event['is_correct'] else 'incorrect'}",
                "",
            ]
        )
    question_lines = [
        f"Problem: {query['problem_text']}",
        f"Correct answer: {query['correct_answer']}",
        "Valid outputs:",
    ]
    for label in candidate_targets:
        question_lines.append(candidate_targets[label])

    seg_instruction = f"{instruction}\n"
    seg_history = "<student_history>\n" + "\n".join(history_lines).rstrip() + "\n</student_history>\n"
    seg_question = "<query>\n" + "\n".join(question_lines) + "\n</query>\nAnswer:"
    prompt = f"{seg_instruction}{seg_history}{STATE_TOKEN}\n{seg_question}"
    return seg_instruction, seg_history, seg_question, prompt


def response_candidates(query: dict[str, Any], answers: dict[str, str]) -> list[str]:
    values = {query["correct_answer"], query.get("malrule_answer") or ""}
    values.update(answers.values())
    return sorted(v for v in values if v != "")


def make_task_rows(raw: dict[str, Any], all_malrule_labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "dataset": "malrulelib_clean",
        "split": raw["split"],
        "student_id": raw["student_id"],
        "history": raw["history"],
        "query": raw["query"],
        "metadata": {
            "raw_example_id": raw["example_id"],
            "malrule_id": raw["malrule_id"],
            "family": raw["family"],
            "skill_id": raw["family"],
            "skill_name": raw["family"],
            "question_id": raw["query"]["surface_id"],
            "is_none_student": raw["malrule_id"] == "NONE",
            "query_answers_by_malrule": raw["query_answers_by_malrule"],
        },
    }

    task_specs: list[tuple[str, str, list[str], dict[str, str]]] = []

    response_labels = response_candidates(raw["query"], raw["query_answers_by_malrule"])
    response_target = raw["query"]["correct_answer"] if raw["malrule_id"] == "NONE" else raw["query"]["malrule_answer"]
    task_specs.append(
        (
            "malrule_next_response",
            response_target,
            response_labels,
            {label: f"<predicted_response>{label}</predicted_response>" for label in response_labels},
        )
    )

    identity_labels = all_malrule_labels
    identity_target = raw["malrule_id"]
    task_specs.append(
        (
            "malrule_identity",
            identity_target,
            identity_labels,
            {label: f"<misconception>{label}</misconception>" for label in identity_labels},
        )
    )

    correctness_labels = ["correct", "incorrect"]
    correctness_target = "correct" if raw["malrule_id"] == "NONE" else "incorrect"
    task_specs.append(
        (
            "malrule_next_correctness",
            correctness_target,
            correctness_labels,
            {label: f"<correctness>{label}</correctness>" for label in correctness_labels},
        )
    )

    for task, target_label, candidate_labels, candidate_targets in task_specs:
        seg_instruction, seg_history, seg_question, prompt = build_segments(
            task,
            raw["history"],
            raw["query"],
            candidate_targets,
        )
        row = dict(base)
        row.update(
            {
                "example_id": f"{raw['example_id']}:{task}",
                "task": task,
                "target_label": target_label,
                "target": candidate_targets[target_label],
                "candidate_labels": candidate_labels,
                "candidate_targets": candidate_targets,
                "seg_instruction": seg_instruction,
                "seg_history": seg_history,
                "seg_question": seg_question,
                "prompt": prompt,
            }
        )
        rows.append(row)
    return rows


def choose_surface(
    malrule_id: str,
    split: str,
    allowed_a: dict[str, list[str]],
    allowed_b: dict[str, list[str]],
    surface_by_id: dict[str, Surface],
    rng: random.Random,
) -> Surface:
    ids = allowed_b[malrule_id] if split == "test_cross_template" else allowed_a[malrule_id]
    return surface_by_id[rng.choice(ids)]


def sample_raw_example(
    split: str,
    idx: int,
    malrules: list[MalruleInfo],
    pooled: list[MalruleInfo],
    cross_capable: list[MalruleInfo],
    allowed_a: dict[str, list[str]],
    allowed_b: dict[str, list[str]],
    surfaces_a: list[Surface],
    surfaces_b: list[Surface],
    surface_by_id: dict[str, Surface],
    rng: random.Random,
    difficulty: Any,
    levels: list[int],
    max_attempts: int,
    history_k: int,
    none_ratio: float,
) -> dict[str, Any]:
    malrule_by_id = {m.malrule_id: m for m in malrules}
    use_none = rng.random() < none_ratio
    if split == "test_cross_template":
        malrule_choices = cross_capable
        surface_pool = surfaces_b
    else:
        malrule_choices = pooled
        surface_pool = surfaces_a

    if not surface_pool:
        raise RuntimeError(f"No surfaces available for split {split}")

    if use_none:
        surface = rng.choice(surface_pool)
        malrule_id = "NONE"
        family = surface.family
        history: list[dict[str, Any]] = []
        for _ in range(history_k):
            item = None
            for _attempt in range(max_attempts):
                try:
                    problem, level = generate_problem(
                        surface.generator_cls,
                        rng.randrange(1_000_000_000),
                        difficulty,
                        levels,
                    )
                    correct = solve_answer(surface.correct_cls, problem)
                    if not is_consistent_with_metadata(problem, correct):
                        continue
                    item = {"problem": problem, "correct_answer": correct, "seed": None, "level": level}
                    break
                except Exception:  # noqa: BLE001
                    continue
            if item is None:
                raise RuntimeError(f"Could not sample NONE history for {surface.surface_id}")
            history.append(problem_record(item, surface, item["correct_answer"], True))

        query_item = None
        for _attempt in range(max_attempts):
            try:
                problem, level = generate_problem(surface.generator_cls, rng.randrange(1_000_000_000), difficulty, levels)
                correct = solve_answer(surface.correct_cls, problem)
                if not is_consistent_with_metadata(problem, correct):
                    continue
                query_item = {"problem": problem, "correct_answer": correct, "seed": None, "level": level}
                break
            except Exception:  # noqa: BLE001
                continue
        if query_item is None:
            raise RuntimeError(f"Could not sample NONE query for {surface.surface_id}")

        query = problem_record(query_item, surface, query_item["correct_answer"], True)
        query["malrule_answer"] = None
        answers = answers_by_family_malrule(query_item, surface, malrules)

    else:
        if not malrule_choices:
            raise RuntimeError(f"No malrules available for split {split}")
        malrule = rng.choice(malrule_choices)
        malrule_id = malrule.malrule_id
        family = malrule.family
        history = []
        for _ in range(history_k):
            surface = choose_surface(malrule_id, split, allowed_a, allowed_b, surface_by_id, rng)
            item = sample_clean_problem(malrule, surface, rng, difficulty, levels, max_attempts)
            if item is None:
                raise RuntimeError(f"Failed to resample allowed pair {malrule_id} x {surface.surface_id}")
            history.append(problem_record(item, surface, item["malrule_answer"], False))

        surface = choose_surface(malrule_id, split, allowed_a, allowed_b, surface_by_id, rng)
        query_item = sample_clean_problem(malrule, surface, rng, difficulty, levels, max_attempts)
        if query_item is None:
            raise RuntimeError(f"Failed to resample query for {malrule_id} x {surface.surface_id}")
        query = problem_record(query_item, surface, query_item["malrule_answer"], False)
        query["malrule_answer"] = query_item["malrule_answer"]
        answers = answers_by_family_malrule(query_item, surface, malrules)

    answer_set = set(answers.values()) | {query["correct_answer"]}
    if query.get("malrule_answer"):
        answer_set.add(query["malrule_answer"])
    distinct_answers = len(answer_set)
    if distinct_answers < 2:
        raise ValueError("Query has fewer than two distinct candidate answers")
    return {
        "example_id": f"malrulelib:{split}:{idx:07d}",
        "student_id": f"malrulelib:{split}:student:{idx:07d}",
        "malrule_id": malrule_id,
        "family": family,
        "history": history,
        "query": query,
        "query_answers_by_malrule": answers,
        "split": split,
        "distinct_query_answers_in_family": distinct_answers,
        "candidate_malrule_count_for_query": len(answers),
    }


def build_split(
    split: str,
    n_raw: int,
    malrules: list[MalruleInfo],
    pooled: list[MalruleInfo],
    cross_capable: list[MalruleInfo],
    allowed_a: dict[str, list[str]],
    allowed_b: dict[str, list[str]],
    surfaces_a: list[Surface],
    surfaces_b: list[Surface],
    surface_by_id: dict[str, Surface],
    rng: random.Random,
    difficulty: Any,
    levels: list[int],
    max_attempts: int,
    history_k: int,
    none_ratio: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failures = 0
    while len(rows) < n_raw:
        try:
            rows.append(
                sample_raw_example(
                    split=split,
                    idx=len(rows),
                    malrules=malrules,
                    pooled=pooled,
                    cross_capable=cross_capable,
                    allowed_a=allowed_a,
                    allowed_b=allowed_b,
                    surfaces_a=surfaces_a,
                    surfaces_b=surfaces_b,
                    surface_by_id=surface_by_id,
                    rng=rng,
                    difficulty=difficulty,
                    levels=levels,
                    max_attempts=max_attempts,
                    history_k=history_k,
                    none_ratio=none_ratio,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            if failures > max(100, n_raw):
                raise RuntimeError(f"Too many sampling failures for {split}: {exc}") from exc
    return rows


def summarize_raw(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_raw": len(rows),
        "malrule_counts": dict(Counter(row["malrule_id"] for row in rows)),
        "family_counts": dict(Counter(row["family"] for row in rows)),
        "none_ratio": sum(row["malrule_id"] == "NONE" for row in rows) / max(1, len(rows)),
        "avg_query_candidate_malrules": sum(row["candidate_malrule_count_for_query"] for row in rows) / max(1, len(rows)),
        "avg_distinct_query_answers": sum(row["distinct_query_answers_in_family"] for row in rows) / max(1, len(rows)),
        "min_distinct_query_answers": min((row["distinct_query_answers_in_family"] for row in rows), default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("experiments/unified_student_c1/data/raw/malrulelib"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/unified_student_c1/data/processed/malrule_c1_clean"))
    parser.add_argument("--families", nargs="+", default=FAMILIES_DEFAULT)
    parser.add_argument("--history-k", type=int, default=4)
    parser.add_argument("--train-raw", type=int, default=3000)
    parser.add_argument("--val-raw", type=int, default=300)
    parser.add_argument("--test-same-raw", type=int, default=300)
    parser.add_argument("--test-cross-raw", type=int, default=300)
    parser.add_argument("--none-ratio", type=float, default=0.18)
    parser.add_argument("--probe-seeds", type=int, default=120)
    parser.add_argument("--max-attempts", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", default=DIFFICULTY_DEFAULT)
    args = parser.parse_args()

    repo = args.repo.resolve()
    rng = random.Random(args.seed)
    difficulty = difficulty_enum(repo, args.difficulty)
    levels = LEVELS_DEFAULT

    malrules = discover_malrules(repo, args.families)
    surfaces, surface_stats = build_surfaces(malrules, difficulty, levels, args.probe_seeds, rng)
    surfaces_a, surfaces_b = split_surfaces(surfaces)
    surface_by_id = {surface.surface_id: surface for surface in surfaces}
    allowed_all, compatibility_stats = find_allowed_surfaces(
        malrules=malrules,
        surfaces=surfaces,
        difficulty=difficulty,
        levels=levels,
        max_attempts=args.max_attempts,
        seed=args.seed + 17,
    )
    surface_a_ids = {surface.surface_id for surface in surfaces_a}
    surface_b_ids = {surface.surface_id for surface in surfaces_b}
    allowed_a = {mid: [sid for sid in sids if sid in surface_a_ids] for mid, sids in allowed_all.items()}
    allowed_b = {mid: [sid for sid in sids if sid in surface_b_ids] for mid, sids in allowed_all.items()}
    pooled = [m for m in malrules if allowed_a.get(m.malrule_id)]
    cross_capable = [m for m in pooled if allowed_b.get(m.malrule_id)]

    all_malrule_labels = ["NONE"] + sorted(m.malrule_id for m in pooled)

    raw_by_split = {
        "train": build_split(
            "train",
            args.train_raw,
            malrules,
            pooled,
            cross_capable,
            allowed_a,
            allowed_b,
            surfaces_a,
            surfaces_b,
            surface_by_id,
            rng,
            difficulty,
            levels,
            args.max_attempts,
            args.history_k,
            args.none_ratio,
        ),
        "val": build_split(
            "val",
            args.val_raw,
            malrules,
            pooled,
            cross_capable,
            allowed_a,
            allowed_b,
            surfaces_a,
            surfaces_b,
            surface_by_id,
            rng,
            difficulty,
            levels,
            args.max_attempts,
            args.history_k,
            args.none_ratio,
        ),
        "test_same_template": build_split(
            "test_same_template",
            args.test_same_raw,
            malrules,
            pooled,
            cross_capable,
            allowed_a,
            allowed_b,
            surfaces_a,
            surfaces_b,
            surface_by_id,
            rng,
            difficulty,
            levels,
            args.max_attempts,
            args.history_k,
            args.none_ratio,
        ),
        "test_cross_template": build_split(
            "test_cross_template",
            args.test_cross_raw,
            malrules,
            pooled,
            cross_capable,
            allowed_a,
            allowed_b,
            surfaces_a,
            surfaces_b,
            surface_by_id,
            rng,
            difficulty,
            levels,
            args.max_attempts,
            args.history_k,
            args.none_ratio,
        ),
    }

    task_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, raw_rows in raw_by_split.items():
        task_rows: list[dict[str, Any]] = []
        for raw in raw_rows:
            task_rows.extend(make_task_rows(raw, all_malrule_labels))
        task_by_split[split] = task_rows

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, raw_rows in raw_by_split.items():
        write_jsonl(args.out_dir / f"raw_{split}.jsonl", raw_rows)
        write_jsonl(args.out_dir / f"{split}.jsonl", task_by_split[split])

    audit_rows = []
    all_raw = [row for rows in raw_by_split.values() for row in rows]
    for row in rng.sample(all_raw, k=min(50, len(all_raw))):
        audit_rows.append(
            {
                "example_id": row["example_id"],
                "split": row["split"],
                "malrule_id": row["malrule_id"],
                "family": row["family"],
                "history": [
                    {
                        "problem_text": h["problem_text"],
                        "student_answer": h["student_answer"],
                        "correct_answer": h["correct_answer"],
                        "is_correct": h["is_correct"],
                        "surface_id": h["surface_id"],
                    }
                    for h in row["history"]
                ],
                "query": {
                    "problem_text": row["query"]["problem_text"],
                    "correct_answer": row["query"]["correct_answer"],
                    "malrule_answer": row["query"].get("malrule_answer"),
                    "surface_id": row["query"]["surface_id"],
                },
                "query_answers_by_malrule": row["query_answers_by_malrule"],
                "distinct_query_answers_in_family": row["distinct_query_answers_in_family"],
            }
        )
    write_json(args.out_dir / "audit_50_samples.json", {"samples": audit_rows})

    stats = {
        "seed": args.seed,
        "families": args.families,
        "history_k": args.history_k,
        "none_ratio_requested": args.none_ratio,
        "malrules_discovered": len(malrules),
        "malrules_pooled": len(pooled),
        "malrules_cross_capable": len(cross_capable),
        "pooled_malrule_ids": sorted(m.malrule_id for m in pooled),
        "cross_capable_malrule_ids": sorted(m.malrule_id for m in cross_capable),
        "surfaces_total": len(surfaces),
        "surfaces_a": len(surfaces_a),
        "surfaces_b": len(surfaces_b),
        "surface_stats": surface_stats,
        "compatibility_stats": compatibility_stats,
        "allowed_a_counts": {mid: len(sids) for mid, sids in sorted(allowed_a.items())},
        "allowed_b_counts": {mid: len(sids) for mid, sids in sorted(allowed_b.items())},
        "splits": {split: summarize_raw(rows) for split, rows in raw_by_split.items()},
        "task_counts": {
            split: dict(Counter(row["task"] for row in rows)) for split, rows in task_by_split.items()
        },
    }
    write_json(args.out_dir / "stats.json", stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

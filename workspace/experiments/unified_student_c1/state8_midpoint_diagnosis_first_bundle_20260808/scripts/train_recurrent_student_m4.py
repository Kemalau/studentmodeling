#!/usr/bin/env python3
"""Train the full unified recurrent student model with causal grounding.

Every unified real-data row is visited exactly once.  A deterministic subset of
micro-steps additionally receives a recurrent donor/native causal pair.  The
decoder sees only the continuous recurrent state and the current query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from contextlib import contextmanager, nullcontext
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from peft.utils import save_and_load as peft_save_and_load
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from recurrent_distillation import topk_teacher_cross_entropy
from distributed_epoch import (
    DistributedContext,
    all_reduce_int,
    barrier,
    close,
    context_from_environment,
    global_position,
    initialize,
    rounds_for_examples,
    synchronize_gradients,
)
from layerwise_recurrent_student_model import LayerwiseRecurrentStudentModel
from recurrent_student_model import RecurrentStudentModel, expand_state_token_alias
from teacher_cache import TeacherCache
from train_recurrent_student_diagnostic import (
    alias_text,
    direct_supervised_batch,
    encode_prompt,
    encode_prompt_target,
    evaluate,
    gradient_audit,
    move,
    read_jsonl,
    sequence_logprob,
    sha256,
    validate_resume_checkpoint,
)


def seed_process_rngs(seed: int, device: torch.device) -> None:
    """Seed CPU and only this process's CUDA device.

    ``torch.manual_seed`` seeds every visible CUDA device.  Under one-process-
    per-GPU MPI that creates a CUDA context from every rank on GPU 0 before
    model loading, consuming enough memory to make the layerwise 32B path OOM.
    The distributed checkpoint already stores CPU and local-device RNG state,
    so rank-local seeding preserves determinism without cross-device contexts.
    """
    seed = int(seed)
    random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(seed)


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean next-token loss over supervised positions without a full FP32 copy.

    Targets use ``-100`` outside the answer span. Selecting those answer rows
    before casting logits to float is exactly equivalent to ignore-index cross
    entropy, but avoids materializing ``sequence_length x vocabulary`` FP32
    logits for long direct-replay examples.
    """
    shifted_labels = labels[:, 1:].contiguous()
    supervised = shifted_labels.ne(-100)
    if not bool(supervised.any()):
        raise ValueError("Causal LM batch has no supervised target tokens")
    selected_logits = logits[:, :-1][supervised].float()
    selected_labels = shifted_labels[supervised]
    return F.cross_entropy(selected_logits, selected_labels)


def history_dropout_at_progress(args, training_progress: float) -> float:
    """Five-stage teacher-to-state curriculum requested by the experiment."""
    if not 0.0 <= float(training_progress) <= 1.0:
        raise ValueError("training_progress must be in [0, 1]")
    if not bool(getattr(args, "history_dropout_curriculum", False)):
        return 1.0
    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    phase = min(int(float(training_progress) * len(levels)), len(levels) - 1)
    return levels[phase]


def dual_path_curriculum_weights(
    args, training_progress: float
) -> tuple[float, float, float]:
    """Return teacher CE, state CE, and online-KD weights for one phase.

    The ordinary dropout curriculum samples one of the two answer routes.  The
    dual-path recipe instead evaluates both routes on every mixed-stage row:
    history dropout becomes a deterministic interpolation coefficient, while
    KD is active only in the three mixed phases.  Endpoint phases therefore
    remain a genuine full-history teacher stage and a genuine state-only
    answer stage.
    """
    dropout = history_dropout_at_progress(args, training_progress)
    if not bool(getattr(args, "dual_path_curriculum", False)):
        return 1.0 - dropout, dropout, 0.0
    kd_profile = 1.0 if 0.0 < dropout < 1.0 else 0.0
    return 1.0 - dropout, dropout, kd_profile


def state_path_gradient_audit_ready(record: dict[str, Any] | None, args) -> bool:
    """Audit every trainable group only after a state-dependent loss is active.

    The first phase of the dual-path curriculum is intentionally full-history
    only: its state CE and answer-logit KD coefficients are both zero.  A
    strict transition-gradient audit on that phase is therefore a false
    failure, not evidence of a detached state path.  Defer the audit until a
    local example actually activates state generation/KD/consistency or a C4
    causal branch.  Once active, ``gradient_audit`` remains fail-closed and
    still requires every present trainable group to have non-zero gradient.
    """
    if record is None:
        return False
    recurrent_state_present = bool(record.get("recurrent_state_present", False))
    state_objective_active = (
        float(record.get("effective_generation_weight", 0.0)) > 0.0
        or float(getattr(args, "behavior_kd_weight", 0.0)) > 0.0
        or (
            float(getattr(args, "online_kd_weight", 0.0)) > 0.0
            and float(record.get("online_kd_phase_weight", 0.0)) > 0.0
        )
        or float(getattr(args, "consistency_weight", 0.0)) > 0.0
    )
    causal_objective_active = (
        record.get("causal_pair_id") is not None
        and float(record.get("causal_effective_weight", 0.0)) > 0.0
    )
    return (recurrent_state_present and state_objective_active) or causal_objective_active


def recurrent_objective_active(
    *,
    recurrent_state_present: bool,
    generation_weight: float,
    behavior_kd_weight: float,
    online_kd_weight: float,
    online_kd_phase_weight: float,
    consistency_weight: float,
) -> bool:
    """Return whether the recurrent branch contributes a differentiable term.

    A teacher-only curriculum phase sets generation and online-KD coefficients
    to zero.  Cold-start rows also have no recurrent state, so their undefined
    consistency term is represented by a detached scalar zero.  Calling
    ``backward`` on that all-zero detached sum is an error; skipping it is the
    exact additive-objective semantics.  Positive consistency counts only when
    a recurrent state exists, while every other positive coefficient remains
    active for both observed-history and cold-start rows.
    """
    weights = (
        generation_weight,
        behavior_kd_weight,
        online_kd_weight * online_kd_phase_weight,
        consistency_weight if recurrent_state_present else 0.0,
    )
    if any(float(weight) < 0.0 for weight in weights):
        raise ValueError("Recurrent objective weights must be non-negative")
    return any(float(weight) > 0.0 for weight in weights)


def detached_teacher_topk(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress online teacher logits at answer positions before graph release."""
    if topk <= 0 or topk > logits.shape[-1]:
        raise ValueError("online KD top-k must be in [1, vocabulary_size]")
    shifted = labels[:, 1:]
    valid = shifted.ne(-100)
    if not bool(valid.any()):
        raise ValueError("Online teacher has no supervised answer positions")
    targets = shifted[valid].detach()
    answer_logits = logits[:, :-1][valid].detach().float()
    values, ids = torch.topk(answer_logits, k=topk, dim=-1)
    return targets, ids, F.log_softmax(values, dim=-1)


def curriculum_progress(
    position: int,
    total_steps: int,
    diagnostic_start_global_position: int | None = None,
    diagnostic_stop_global_position: int | None = None,
) -> float:
    """Normalize over the active segment, not the 259k-row source epoch."""
    start = int(diagnostic_start_global_position or 0)
    stop = (
        int(diagnostic_stop_global_position)
        if diagnostic_stop_global_position is not None
        else int(total_steps)
    )
    if not start < stop <= int(total_steps):
        raise ValueError("Curriculum segment boundaries are invalid")
    relative = max(0, int(position) - start)
    segment_total = stop - start
    return min(relative / max(segment_total - 1, 1), 1.0)


@contextmanager
def seeded_history_dropout(seed: int):
    """Isolate curriculum sampling from the checkpointed training RNG stream."""
    state = random.getstate()
    random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(state)


def load_replicated_data_parallel_adapter(
    model: torch.nn.Module, adapter_source: Path
) -> PeftModel:
    """Load a full adapter replica without PEFT's tensor-parallel sharding.

    M4 keeps a complete quantized model on every GPU and synchronizes gradients
    explicitly. PEFT 0.19.1 otherwise enters its tensor-parallel checkpoint
    path whenever torch.distributed is initialized, even though this model has
    no tensor-parallel device mesh. Fail closed if a real TP mesh is introduced.
    """
    meshed = [
        name
        for name, module in model.named_modules()
        if getattr(module, "_hf_device_mesh", None) is not None
    ]
    if meshed:
        raise ValueError(
            "Replicated adapter loading cannot be used with a tensor-parallel "
            f"device mesh: {meshed[:5]}"
        )
    original = peft_save_and_load._maybe_shard_state_dict_for_tp
    peft_save_and_load._maybe_shard_state_dict_for_tp = lambda *args, **kwargs: None
    try:
        return PeftModel.from_pretrained(
            model, str(adapter_source), is_trainable=True
        )
    finally:
        peft_save_and_load._maybe_shard_state_dict_for_tp = original


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_file(path: Path) -> None:
    """Force a completed artifact through the mounted Blob writeback cache."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    """Persist directory-entry publication where the mounted filesystem allows it."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_file(temporary: Path, final: Path) -> None:
    """Publish one fully durable file, then verify the final entry is readable."""
    fsync_file(temporary)
    temporary.replace(final)
    fsync_directory(final.parent)
    if not final.is_file() or final.stat().st_size <= 0:
        raise IOError(f"Published artifact is missing or empty: {final}")


def publish_text(temporary: Path, final: Path, text: str) -> None:
    temporary.write_text(text, encoding="utf-8")
    publish_file(temporary, final)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: str,
    total_steps: int,
    warmup_ratio: float,
):
    if total_steps <= 0:
        raise ValueError("Scheduler total_steps must be positive")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if schedule != "cosine":
        raise ValueError(f"Unknown LR schedule: {schedule}")
    warmup_steps = round(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def build_epoch_schedule(
    real_count: int, causal_count: int, seed: int
) -> tuple[list[int], dict[int, int]]:
    """Shuffle real rows once and spread every causal pair over distinct steps."""
    if real_count <= 0:
        raise ValueError("real_count must be positive")
    if causal_count < 0 or causal_count > real_count:
        raise ValueError("causal_count must be in [0, real_count]")
    rng = random.Random(seed)
    real_order = list(range(real_count))
    causal_order = list(range(causal_count))
    rng.shuffle(real_order)
    rng.shuffle(causal_order)
    if causal_count == 0:
        return real_order, {}
    positions = [int(index * real_count / causal_count) for index in range(causal_count)]
    if len(set(positions)) != causal_count:
        raise AssertionError("Causal schedule positions are not unique")
    return real_order, dict(zip(positions, causal_order, strict=True))


def build_repeated_causal_schedule(
    real_count: int, causal_count: int, fraction: float, seed: int
) -> dict[int, int]:
    """Spread deterministic causal rehearsal over a real-data epoch."""
    if real_count <= 0 or causal_count <= 0:
        raise ValueError("real_count and causal_count must be positive")
    if not 0 < fraction <= 1:
        raise ValueError("causal rehearsal fraction must be in (0, 1]")
    exposures = max(causal_count, round(real_count * fraction))
    if exposures > real_count:
        raise ValueError("Causal exposures cannot exceed real-data steps")
    positions = [int(index * real_count / exposures) for index in range(exposures)]
    if len(set(positions)) != exposures:
        raise AssertionError("Repeated causal schedule positions are not unique")
    rng = random.Random(seed)
    causal_order = []
    while len(causal_order) < exposures:
        cycle = list(range(causal_count))
        rng.shuffle(cycle)
        causal_order.extend(cycle)
    return dict(zip(positions, causal_order[:exposures], strict=True))


def causal_exposure_multiplier(
    real_count: int, exposure_count: int, normalize: bool
) -> float:
    """Scale scheduled causal examples to an average causal objective.

    Distributed gradient synchronization averages all branches by the number
    of real examples.  Without this correction, a causal branch scheduled on
    fraction ``p`` of the corpus has effective coefficient ``p * weight`` and
    changes again when the replay fraction changes.  Multiplying each causal
    exposure by ``N_real / N_causal`` makes the epoch-level objective
    ``mean(real losses) + weight * mean(causal losses)``.
    """
    if real_count <= 0 or exposure_count <= 0:
        raise ValueError("real_count and exposure_count must be positive")
    return float(real_count) / float(exposure_count) if normalize else 1.0


def load_diagnostic_index_manifest(
    path: Path,
    train_rows: list[dict[str, Any]],
    *,
    source_train_sha256: str,
    start_global_position: int,
) -> list[int]:
    """Validate a diagnostic-only row-index curriculum against its source pool."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "layerwise_task_retention_index_manifest_v1":
        raise ValueError("Diagnostic index-manifest stage drift")
    if manifest.get("source_train_sha256") != source_train_sha256:
        raise ValueError("Diagnostic index-manifest source hash drift")
    if int(manifest.get("start_global_position", -1)) != int(start_global_position):
        raise ValueError("Diagnostic index-manifest start boundary drift")
    raw_indices = manifest.get("indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("Diagnostic index manifest has no indices")
    if any(type(index) is not int for index in raw_indices):
        raise ValueError("Diagnostic index manifest contains non-integer indices")
    indices = list(raw_indices)
    if len(set(indices)) != len(indices):
        raise ValueError("Diagnostic index manifest contains duplicate indices")
    if any(index < 0 or index >= len(train_rows) for index in indices):
        raise ValueError("Diagnostic index manifest contains out-of-range indices")
    observed_tasks = Counter(str(train_rows[index]["task"]) for index in indices)
    observed_sources = Counter(str(train_rows[index]["source"]) for index in indices)
    if dict(sorted(observed_tasks.items())) != manifest.get("task_counts"):
        raise ValueError("Diagnostic index-manifest task counts drift")
    if dict(sorted(observed_sources.items())) != manifest.get("source_counts"):
        raise ValueError("Diagnostic index-manifest source counts drift")
    return indices


def load_promoted_real_order_manifest(
    path: Path,
    train_rows: list[dict[str, Any]],
    *,
    source_train_sha256: str,
    checkpoint_payload: dict[str, Any],
) -> list[int]:
    """Validate a promoted, full-epoch order against an exact resume checkpoint.

    A task-retention rehearsal may identify a better short curriculum, but its
    diagnostic rows cannot simply be appended to the main run: some may have
    already appeared in the completed prefix.  The promotion builder replaces
    those rows with support-matched unseen examples and emits a complete
    permutation.  This loader fails closed unless that permutation preserves
    every already-consumed position and every row still appears exactly once.
    """

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "layerwise_task_retention_promoted_order_v1":
        raise ValueError("Promoted real-order manifest stage drift")
    if manifest.get("source_train_sha256") != source_train_sha256:
        raise ValueError("Promoted real-order manifest source hash drift")
    if manifest.get("one_epoch_permutation") is not True:
        raise ValueError("Promoted real-order manifest is not a one-epoch permutation")
    if manifest.get("completed_prefix_unchanged") is not True:
        raise ValueError("Promoted real-order manifest does not preserve the prefix")
    if int(manifest.get("formal_example_id_overlap", -1)) != 0:
        raise ValueError("Promoted real-order manifest overlaps formal evaluation")

    if checkpoint_payload.get("distributed") is not True:
        raise ValueError("Promoted real order requires a distributed checkpoint")
    world_size = int(checkpoint_payload.get("world_size", -1))
    round_index = int(checkpoint_payload.get("round", -1))
    prefix_count = world_size * round_index
    if world_size <= 1 or round_index <= 0:
        raise ValueError("Promoted real-order checkpoint boundary is invalid")
    if int(manifest.get("source_world_size", -1)) != world_size:
        raise ValueError("Promoted real-order source world-size drift")
    source_round = int(manifest.get("source_checkpoint_round", -1))
    source_prefix_count = int(manifest.get("completed_prefix_rows", -1))
    manifest_sha = file_sha256(path)
    checkpoint_config = checkpoint_payload.get("config") or {}
    continuing_promotion = (
        checkpoint_config.get("promoted_real_order_applied") is True
        and checkpoint_config.get("promoted_real_order_manifest_sha256") == manifest_sha
    )
    if continuing_promotion:
        if source_round <= 0 or round_index < source_round:
            raise ValueError("Promoted real-order continuation precedes its source")
        if source_prefix_count != source_round * world_size:
            raise ValueError("Promoted real-order source-prefix boundary drift")
    else:
        if source_round != round_index:
            raise ValueError("Promoted real-order source checkpoint-round drift")
        if source_prefix_count != prefix_count:
            raise ValueError("Promoted real-order completed-prefix boundary drift")

    raw_order = manifest.get("real_order")
    if not isinstance(raw_order, list) or any(type(index) is not int for index in raw_order):
        raise ValueError("Promoted real-order manifest has invalid indices")
    order = list(raw_order)
    row_count = len(train_rows)
    if int(manifest.get("epoch_rows", -1)) != row_count:
        raise ValueError("Promoted real-order epoch support drift")
    if len(order) != row_count or set(order) != set(range(row_count)):
        raise ValueError("Promoted real order is not a complete permutation")
    observed_order_sha = hashlib.sha256("\n".join(map(str, order)).encode()).hexdigest()
    if manifest.get("real_order_sha256") != observed_order_sha:
        raise ValueError("Promoted real-order SHA256 drift")
    checkpoint_order = list(checkpoint_payload.get("real_order") or [])
    if len(checkpoint_order) != row_count:
        raise ValueError("Resume checkpoint real order is incomplete")
    if continuing_promotion:
        if order != checkpoint_order:
            raise ValueError("Promoted real-order continuation changed its epoch order")
    elif order[:prefix_count] != checkpoint_order[:prefix_count]:
        raise ValueError("Promoted real order changes the completed prefix")

    history = list(checkpoint_payload.get("history") or [])
    if len(history) != prefix_count:
        raise ValueError("Resume checkpoint history does not prove the completed prefix")
    positions = [int(row.get("global_position", -1)) for row in history]
    if positions != list(range(prefix_count)):
        raise ValueError("Resume checkpoint history coordinates are not contiguous")
    expected_ids = [str(train_rows[index]["example_id"]) for index in order[:prefix_count]]
    observed_ids = [str(row.get("example_id", "")) for row in history]
    if expected_ids != observed_ids:
        raise ValueError("Promoted real-order prefix identities do not match checkpoint history")
    promoted_indices = manifest.get("promoted_indices")
    if not isinstance(promoted_indices, list) or any(
        type(index) is not int for index in promoted_indices
    ):
        raise ValueError("Promoted real-order curriculum indices are invalid")
    promoted_rows = int(manifest.get("promoted_rows", -1))
    if promoted_rows != len(promoted_indices) or promoted_rows <= 0:
        raise ValueError("Promoted real-order curriculum support drift")
    if order[source_prefix_count : source_prefix_count + promoted_rows] != promoted_indices:
        raise ValueError("Promoted real-order curriculum is not at the resume boundary")
    promoted_ids = [str(train_rows[index]["example_id"]) for index in promoted_indices]
    observed_promoted_id_sha = hashlib.sha256("\n".join(promoted_ids).encode()).hexdigest()
    if manifest.get("promoted_example_ids_sha256") != observed_promoted_id_sha:
        raise ValueError("Promoted real-order example-ID SHA256 drift")
    return order


def paired_preference_loss(
    native_gold_nll: torch.Tensor,
    native_donor_nll: torch.Tensor,
    donor_gold_nll: torch.Tensor,
    donor_native_nll: torch.Tensor,
    margin: float,
    *,
    preference_nlls: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]
    | None = None,
    objective_mode: str = "symmetric",
    native_margin_weight: float = 1.0,
    donor_margin_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve B under native state and prefer A only under donor state.

    ``symmetric`` retains the preregistered difference-in-differences hinge.
    ``directional`` applies a hinge to each state separately.  Unequal
    directional weights are useful only as a short bootstrap: when donor and
    native states are initially identical, the two symmetric preference
    gradients are exact opposites and can cancel before the updater learns to
    separate the states.
    """
    if margin < 0:
        raise ValueError("Causal margin must be non-negative")
    if native_margin_weight < 0 or donor_margin_weight < 0:
        raise ValueError("Directional margin weights must be non-negative")
    if objective_mode not in {"symmetric", "directional"}:
        raise ValueError(f"Unknown causal objective mode: {objective_mode}")
    preference_values = preference_nlls or (
        native_gold_nll,
        native_donor_nll,
        donor_gold_nll,
        donor_native_nll,
    )
    pref_native_gold, pref_native_donor, pref_donor_gold, pref_donor_native = (
        preference_values
    )
    native_advantage = pref_native_donor - pref_native_gold
    donor_advantage = pref_donor_native - pref_donor_gold
    if objective_mode == "symmetric":
        preference = 0.5 * (native_advantage + donor_advantage)
        margin_loss = F.relu(preference.new_tensor(float(margin)) - preference)
    else:
        weight_sum = float(native_margin_weight + donor_margin_weight)
        if weight_sum <= 0:
            raise ValueError("At least one directional margin weight must be positive")
        native_hinge = F.relu(
            native_advantage.new_tensor(float(margin)) - native_advantage
        )
        donor_hinge = F.relu(
            donor_advantage.new_tensor(float(margin)) - donor_advantage
        )
        margin_loss = (
            float(native_margin_weight) * native_hinge
            + float(donor_margin_weight) * donor_hinge
        ) / weight_sum
    gold_loss = 0.5 * (native_gold_nll + donor_gold_nll)
    return gold_loss + margin_loss, margin_loss


def paired_preference_split_coefficients(
    native_advantage: float,
    donor_advantage: float,
    margin: float,
    *,
    objective_mode: str,
    native_margin_weight: float = 1.0,
    donor_margin_weight: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return linear branch coefficients for an exact two-view backward split.

    Away from the hinge boundary, ReLU is linear. Freezing its activation from
    detached advantages in the deterministic causal eval-mode objective lets
    the native and donor graphs be
    backpropagated and released separately without changing the scalar loss or
    its gradient.  Each branch is ``0.5 * gold + c * advantage + k``.
    """
    if margin < 0:
        raise ValueError("Causal margin must be non-negative")
    if objective_mode == "symmetric":
        active = float(
            0.5 * (float(native_advantage) + float(donor_advantage)) < margin
        )
        return (
            -0.5 * active,
            0.5 * float(margin) * active,
            -0.5 * active,
            0.5 * float(margin) * active,
        )
    if objective_mode != "directional":
        raise ValueError(f"Unknown causal objective mode: {objective_mode}")
    if native_margin_weight < 0 or donor_margin_weight < 0:
        raise ValueError("Directional margin weights must be non-negative")
    weight_sum = float(native_margin_weight + donor_margin_weight)
    if weight_sum <= 0:
        raise ValueError("At least one directional margin weight must be positive")
    native_scale = float(native_margin_weight) / weight_sum
    donor_scale = float(donor_margin_weight) / weight_sum
    native_active = float(float(native_advantage) < margin)
    donor_active = float(float(donor_advantage) < margin)
    return (
        -native_scale * native_active,
        native_scale * float(margin) * native_active,
        -donor_scale * donor_active,
        donor_scale * float(margin) * donor_active,
    )


def norm_matched_random_state(reference: torch.Tensor, seed: int) -> torch.Tensor:
    """Match the frozen C4 isotropic control without leaking its RNG to training.

    The reference norm is detached deliberately: the random-negative branch
    trains the decoder to reject an unstructured state direction, but it must
    not create an incentive to shrink or inflate the native recurrent state.
    Generation happens on CPU exactly as in ``evaluate_recurrent_causal_gate``
    so CUDA rank/device RNG differences cannot change the control.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    reference_float = reference.detach().float().cpu()
    noise_norm = torch.linalg.vector_norm(noise, dim=-1, keepdim=True).clamp_min(1e-12)
    reference_norm = torch.linalg.vector_norm(
        reference_float, dim=-1, keepdim=True
    )
    return (noise * (reference_norm / noise_norm)).to(
        device=reference.device, dtype=reference.dtype
    )


def specificity_triplet_loss(
    native_gold_nll: torch.Tensor,
    native_donor_nll: torch.Tensor,
    donor_gold_nll: torch.Tensor,
    donor_native_nll: torch.Tensor,
    random_native_nll: torch.Tensor,
    random_donor_nll: torch.Tensor,
    margin: float,
    *,
    preference_nlls: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Align training with the frozen ``swap - random`` specificity contrast.

    Native and norm-matched random states should prefer recipient response B;
    the matched donor state should prefer donor response A.  The fourth hinge
    directly separates the donor A-vs-B margin from the random A-vs-B margin,
    which is the continuous analogue of the frozen C4 ``D_specific`` gate.
    """
    if margin < 0:
        raise ValueError("Causal margin must be non-negative")
    preference_values = preference_nlls or (
        native_gold_nll,
        native_donor_nll,
        donor_gold_nll,
        donor_native_nll,
        random_native_nll,
        random_donor_nll,
    )
    (
        pref_native_gold,
        pref_native_donor,
        pref_donor_gold,
        pref_donor_native,
        pref_random_native,
        pref_random_donor,
    ) = preference_values
    native_advantage = pref_native_donor - pref_native_gold
    donor_advantage = pref_donor_native - pref_donor_gold
    random_native_advantage = pref_random_donor - pref_random_native
    specificity_advantage = donor_advantage + random_native_advantage
    margin_tensor = native_advantage.new_tensor(float(margin))
    margin_loss = torch.stack(
        (
            F.relu(margin_tensor - native_advantage),
            F.relu(margin_tensor - donor_advantage),
            F.relu(margin_tensor - random_native_advantage),
            F.relu(2.0 * margin_tensor - specificity_advantage),
        )
    ).mean()
    gold_loss = (
        native_gold_nll + donor_gold_nll + random_native_nll
    ) / 3.0
    diagnostics = {
        "native_advantage": native_advantage,
        "donor_advantage": donor_advantage,
        "random_native_advantage": random_native_advantage,
        "specificity_advantage": specificity_advantage,
    }
    return gold_loss + margin_loss, margin_loss, diagnostics


def specificity_triplet_branch_losses(
    native_losses: torch.Tensor,
    native_preferences: torch.Tensor,
    donor_losses: torch.Tensor,
    donor_preferences: torch.Tensor,
    random_losses: torch.Tensor,
    random_preferences: torch.Tensor,
    margin: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Split the exact triplet scalar into three independent two-view graphs."""
    for name, losses, preferences in (
        ("native", native_losses, native_preferences),
        ("donor", donor_losses, donor_preferences),
        ("random", random_losses, random_preferences),
    ):
        if losses.shape[0] != 2 or preferences.shape[0] != 2:
            raise ValueError(f"Specificity {name} branch requires two aligned views")
    if margin < 0:
        raise ValueError("Causal margin must be non-negative")
    native_advantage = native_preferences[1] - native_preferences[0]
    donor_advantage = donor_preferences[1] - donor_preferences[0]
    random_native_advantage = random_preferences[1] - random_preferences[0]
    specificity_advantage = donor_advantage + random_native_advantage
    margin_tensor = native_advantage.new_tensor(float(margin))
    native_active = (native_advantage.detach() < float(margin)).to(
        dtype=native_advantage.dtype
    )
    donor_active = (donor_advantage.detach() < float(margin)).to(
        dtype=native_advantage.dtype
    )
    random_active = (random_native_advantage.detach() < float(margin)).to(
        dtype=native_advantage.dtype
    )
    specificity_active = (specificity_advantage.detach() < 2.0 * float(margin)).to(
        dtype=native_advantage.dtype
    )
    native_margin = native_active * (margin_tensor - native_advantage) / 4.0
    donor_margin = (
        (donor_active + specificity_active)
        * (margin_tensor - donor_advantage)
        / 4.0
    )
    random_margin = (
        (random_active + specificity_active)
        * (margin_tensor - random_native_advantage)
        / 4.0
    )
    native_loss = native_losses[0] / 3.0 + native_margin
    donor_loss = donor_losses[0] / 3.0 + donor_margin
    random_loss = random_losses[0] / 3.0 + random_margin
    full_margin = native_margin + donor_margin + random_margin
    return native_loss, donor_loss, random_loss, full_margin, {
        "native_advantage": native_advantage,
        "donor_advantage": donor_advantage,
        "random_native_advantage": random_native_advantage,
        "specificity_advantage": specificity_advantage,
    }


def causal_lm_loss_per_sequence(
    logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(-100)
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("Every sequence must have at least one supervised target token")
    positions = valid.nonzero(as_tuple=False)
    selected_logits = logits[:, :-1][valid].float()
    selected_labels = shifted_labels[valid]
    selected_loss = F.cross_entropy(
        selected_logits, selected_labels, reduction="none"
    )
    totals = selected_loss.new_zeros(logits.shape[0])
    totals.scatter_add_(0, positions[:, 0], selected_loss)
    if reduction == "sum":
        return totals
    if reduction == "mean":
        return totals / valid.sum(dim=-1)
    raise ValueError(f"Unknown per-sequence reduction: {reduction}")


def pad_supervised_batches(
    batches: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Combine batch-size-one supervised records without changing target masks."""
    if not batches:
        raise ValueError("Cannot pad an empty supervised batch list")
    device = batches[0]["input_ids"].device
    max_len = max(int(batch["input_ids"].shape[1]) for batch in batches)
    input_ids, attention, labels, prompt_ends = [], [], [], []
    for batch in batches:
        length = int(batch["input_ids"].shape[1])
        pad = max_len - length
        input_ids.append(
            F.pad(batch["input_ids"], (0, pad), value=int(pad_token_id))
        )
        attention.append(F.pad(batch["attention_mask"], (0, pad), value=0))
        labels.append(F.pad(batch["labels"], (0, pad), value=-100))
        prompt_ends.append(batch["prompt_ends"])
    return {
        "input_ids": torch.cat(input_ids, dim=0).to(device),
        "attention_mask": torch.cat(attention, dim=0).to(device),
        "labels": torch.cat(labels, dim=0).to(device),
        "prompt_ends": torch.cat(prompt_ends, dim=0).to(device),
    }


def summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    ordinary = defaultdict(list)
    causal = []
    for row in history:
        ordinary[(row["task"], row["source"])].append(row)
        if row.get("causal_pair_id") is not None:
            causal.append(row)
    fields = (
        "loss",
        "generation_loss",
        "behavior_kd_loss",
        "consistency_loss",
        "direct_loss",
        "consistency_cosine_loss",
        "consistency_relative_mse",
        "causal_trajectory_loss",
    )
    by_task_source = {
        f"{task}/{source}": {
            "n": len(rows),
            **{
                f"mean_{field}": sum(float(row[field]) for row in rows) / len(rows)
                if all(field in row for row in rows)
                else 0.0
                for field in fields
            },
        }
        for (task, source), rows in sorted(ordinary.items())
    }
    causal_summary = {
        "n": len(causal),
        "mean_causal_loss": (
            sum(float(row["causal_loss"]) for row in causal) / len(causal)
            if causal
            else 0.0
        ),
        "mean_causal_margin_loss": (
            sum(float(row["causal_margin_loss"]) for row in causal) / len(causal)
            if causal
            else 0.0
        ),
        "mean_causal_trajectory_loss": (
            sum(float(row.get("causal_trajectory_loss", 0.0)) for row in causal)
            / len(causal)
            if causal
            else 0.0
        ),
    }
    diagnostic_fields = (
        "causal_native_advantage",
        "causal_donor_advantage",
        "causal_preference",
        "causal_state_cosine",
        "causal_state_delta_norm",
        "causal_state_relative_delta",
        "causal_random_native_advantage",
        "causal_specificity_advantage",
    )
    for field in diagnostic_fields:
        values = [float(row[field]) for row in causal if field in row]
        if values:
            causal_summary[f"mean_{field}"] = sum(values) / len(values)
    phases = defaultdict(int)
    for row in causal:
        if row.get("causal_objective_phase") is not None:
            phases[str(row["causal_objective_phase"])] += 1
    if phases:
        causal_summary["objective_phase_counts"] = dict(sorted(phases.items()))
    return {"by_task_source": by_task_source, "causal": causal_summary}


def audit_history_integrity(
    history: list[dict[str, Any]], *, expected_start: int = 0
) -> dict[str, Any]:
    """Return fail-closed evidence that the gathered epoch has no holes.

    Counting unique example IDs alone cannot distinguish a genuine exact epoch
    from a malformed distributed gather with duplicate/missing positions.  The
    global coordinate is assigned before every rank-local forward pass, so the
    authoritative history must be exactly the contiguous interval starting at
    zero, with one unique example ID per coordinate.
    """

    positions: list[int] = []
    example_ids: list[str] = []
    for index, row in enumerate(history):
        if "global_position" not in row or "example_id" not in row:
            raise ValueError(f"Training history row {index} lacks identity fields")
        positions.append(int(row["global_position"]))
        example_ids.append(str(row["example_id"]))
    rows = len(history)
    unique_positions = len(set(positions))
    unique_examples = len(set(example_ids))
    contiguous = positions == list(range(expected_start, expected_start + rows))
    return {
        "history_rows": rows,
        "history_global_position_start": positions[0] if positions else None,
        "history_global_position_end": positions[-1] if positions else None,
        "history_unique_global_positions": unique_positions,
        "history_global_positions_contiguous": contiguous,
        "history_unique_examples": unique_examples,
        "history_integrity_pass": (
            contiguous
            and unique_positions == rows
            and unique_examples == rows
        ),
    }


def recurrent_state_payload(wrapper, config: dict[str, Any]) -> dict[str, Any]:
    """Serialize architecture-specific recurrent modules fail-closed."""
    payload: dict[str, Any] = {"config": config}
    if isinstance(wrapper, RecurrentStudentModel):
        payload.update(
            {
                "state_in_proj": wrapper.state_in_proj.state_dict(),
                "state_out_proj": wrapper.state_out_proj.state_dict(),
            }
        )
    elif not isinstance(wrapper, LayerwiseRecurrentStudentModel):
        raise TypeError(f"Unsupported recurrent wrapper: {type(wrapper).__name__}")
    else:
        payload.update(
            {
                "transition_in": wrapper.transition_in.state_dict(),
                "transition_out": wrapper.transition_out.state_dict(),
            }
        )
    return payload


def load_recurrent_state_payload(wrapper, payload: dict[str, Any]) -> None:
    """Restore only state modules owned by the selected architecture."""
    config = payload.get("config") or {}
    expected = "layerwise" if isinstance(wrapper, LayerwiseRecurrentStudentModel) else "scalar"
    recorded = str(config.get("state_architecture", "scalar"))
    if recorded != expected:
        raise ValueError(
            f"Recurrent-state architecture mismatch: expected={expected} recorded={recorded}"
        )
    if isinstance(wrapper, RecurrentStudentModel):
        wrapper.state_in_proj.load_state_dict(payload["state_in_proj"])
        wrapper.state_out_proj.load_state_dict(payload["state_out_proj"])
    else:
        source_rank = int(config.get("layerwise_transition_rank", 0))
        if source_rank != wrapper.transition_rank:
            raise ValueError(
                "Layerwise transition rank mismatch: "
                f"expected={wrapper.transition_rank} recorded={source_rank}"
            )
        wrapper.transition_in.load_state_dict(payload["transition_in"])
        wrapper.transition_out.load_state_dict(payload["transition_out"])


RESUME_OPERATIONAL_CONFIG_FIELDS = frozenset(
    {
        # These fields may change when a service-limited attempt is resumed;
        # none changes the optimizer trajectory or the scientific objective.
        "checkpoint_retention",
        "checkpoint_mirror_dir",
        "distributed_rounds",
        "effective_init_adapter",
        "output_dir",
        "resume_from_checkpoint",
        "save_steps",
        # Entrypoint source is pinned independently by each AMLT launcher.
        # Exact recovery may patch checkpoint plumbing without changing any
        # science-bearing config field; the current wrapper digest therefore
        # belongs to the recovery attempt, not to the frozen optimizer recipe.
        "entrypoint_sha256",
        "direct_offload_min_tokens",
        "diagnostic_real_index",
        "diagnostic_warm_start_checkpoint",
        "diagnostic_start_global_position",
        "diagnostic_index_manifest",
        "diagnostic_index_manifest_sha256",
        # A promoted order is allowed only after the dedicated fail-closed
        # manifest validator proves that the consumed prefix is byte-for-byte
        # unchanged and the full order is still a one-epoch permutation.
        "promoted_real_order_manifest",
        "promoted_real_order_manifest_sha256",
        "promoted_real_order_applied",
        "promoted_real_order_source_round",
    }
)


def _canonical_config_value(value: Any) -> Any:
    """Normalize argparse/path containers before resume-recipe comparison."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _canonical_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_config_value(item) for item in value]
    return value


def validate_distributed_resume_recipe(
    checkpoint_config: dict[str, Any], current_config: dict[str, Any]
) -> None:
    """Reject scientific recipe drift before loading optimizer state.

    Distributed recovery is allowed to change only checkpoint plumbing.  Data,
    state architecture/token, loss weights, curriculum, optimizer schedule,
    seed, world size, and exact-epoch plan must match the atomic checkpoint.
    """
    if not isinstance(checkpoint_config, dict) or not checkpoint_config:
        raise ValueError("Distributed resume checkpoint lacks its training config")
    keys = (set(checkpoint_config) | set(current_config)) - set(
        RESUME_OPERATIONAL_CONFIG_FIELDS
    )
    missing = object()
    drift: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        before = checkpoint_config.get(key, missing)
        after = current_config.get(key, missing)
        if before is missing or after is missing:
            # Adding an optional argparse field with default ``None`` is not a
            # scientific recipe change.  This narrowly scoped compatibility
            # rule keeps old exact checkpoints resumable while still rejecting
            # every missing/non-null transition (for example enabling a new
            # cache or changing its bound manifest mid-run).
            present = after if before is missing else before
            if present is None:
                continue
            drift[key] = {
                "checkpoint": "<missing>" if before is missing else _canonical_config_value(before),
                "current": "<missing>" if after is missing else _canonical_config_value(after),
            }
            continue
        canonical_before = _canonical_config_value(before)
        canonical_after = _canonical_config_value(after)
        if canonical_before != canonical_after:
            drift[key] = {
                "checkpoint": canonical_before,
                "current": canonical_after,
            }
    if drift:
        raise ValueError(
            "Distributed resume scientific recipe drift: "
            + json.dumps(drift, sort_keys=True, default=str)
        )


def save_checkpoint(
    wrapper: RecurrentStudentModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    output_dir: Path,
    step: int,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    real_order: list[int],
    causal_schedule: dict[int, int],
) -> None:
    checkpoint = output_dir / f"checkpoint-{step}"
    adapter_dir = checkpoint / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    wrapper.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_model = adapter_dir / "adapter_model.safetensors"
    fsync_file(adapter_model)
    fsync_directory(adapter_dir)
    payload = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
        "real_order": real_order,
        "causal_schedule": causal_schedule,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state_all": torch.cuda.get_rng_state_all(),
        "config": config,
        **recurrent_state_payload(wrapper, config),
    }
    temporary = checkpoint / "recurrent_state.pt.partial"
    torch.save(payload, temporary)
    recurrent_state = checkpoint / "recurrent_state.pt"
    publish_file(temporary, recurrent_state)
    receipt = {
        "status": "complete",
        "step": step,
        "sha256": {
            "adapter_model": sha256(adapter_model),
            "recurrent_state": sha256(recurrent_state),
        },
    }
    temporary_receipt = checkpoint / "checkpoint_completion.json.partial"
    publish_text(
        temporary_receipt,
        checkpoint / "checkpoint_completion.json",
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )


def save_distributed_checkpoint(
    wrapper: RecurrentStudentModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    output_dir: Path,
    round_index: int,
    config: dict[str, Any],
    local_history: list[dict[str, Any]],
    history_base: list[dict[str, Any]],
    real_order: list[int],
    causal_schedule: dict[int, int],
    context: DistributedContext,
    retention: int = 0,
    mirror_dir: Path | None = None,
    preserve_rounds: tuple[int, ...] = (),
) -> None:
    """Save one synchronized checkpoint with every rank's RNG and history."""
    if not context.enabled:
        raise ValueError("Distributed checkpoint requires a distributed context")
    rng = {
        "rank": context.rank,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state(context.local_rank),
    }
    gathered_rng = [None] * context.world_size if context.is_main else None
    gathered_history = [None] * context.world_size if context.is_main else None
    dist.gather_object(rng, gathered_rng, dst=0)
    dist.gather_object(local_history, gathered_history, dst=0)
    if context.is_main:
        checkpoint = output_dir / f"checkpoint-round-{round_index}"
        adapter_dir = checkpoint / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        wrapper.backbone.save_pretrained(adapter_dir, safe_serialization=True)
        adapter_model = adapter_dir / "adapter_model.safetensors"
        fsync_file(adapter_model)
        fsync_directory(adapter_dir)
        combined_history = list(history_base)
        for rank_history in gathered_history or []:
            combined_history.extend(rank_history or [])
        combined_history.sort(key=lambda row: int(row["global_position"]))
        payload = {
            "distributed": True,
            "world_size": context.world_size,
            "round": round_index,
            "step": round_index,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": combined_history,
            "real_order": real_order,
            "causal_schedule": causal_schedule,
            "rank_rng_states": gathered_rng,
            "config": config,
            **recurrent_state_payload(wrapper, config),
        }
        temporary = checkpoint / "recurrent_state.pt.partial"
        torch.save(payload, temporary)
        recurrent_state = checkpoint / "recurrent_state.pt"
        publish_file(temporary, recurrent_state)
        receipt = {
            "status": "complete",
            "round": round_index,
            "world_size": context.world_size,
            "sha256": {
                "adapter_model": sha256(adapter_model),
                "recurrent_state": sha256(recurrent_state),
            },
        }
        temporary_receipt = checkpoint / "checkpoint_completion.json.partial"
        publish_text(
            temporary_receipt,
            checkpoint / "checkpoint_completion.json",
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        )
        persisted = json.loads(
            (checkpoint / "checkpoint_completion.json").read_text(encoding="utf-8")
        )
        if persisted != receipt:
            raise IOError(f"Checkpoint receipt verification failed: {checkpoint}")
        if mirror_dir is not None:
            mirror_distributed_checkpoint(checkpoint, mirror_dir)
        prune_distributed_checkpoints(
            output_dir, retention=retention, preserve_rounds=preserve_rounds
        )
    barrier(context)


def mirror_distributed_checkpoint(checkpoint: Path, mirror_dir: Path) -> Path:
    """Publish a complete checkpoint to shared storage with the receipt last.

    A service retry may observe an interrupted copy on a Blob/FUSE mount.  Such
    a directory remains ineligible for resume because it has no completion
    receipt.  A verified receipt is copied only after the adapter and recurrent
    payload match the source hashes.
    """
    source_receipt = checkpoint / "checkpoint_completion.json"
    source_state = checkpoint / "recurrent_state.pt"
    source_adapter = checkpoint / "adapter/adapter_model.safetensors"
    if not source_receipt.is_file():
        raise FileNotFoundError(f"Checkpoint receipt is missing: {source_receipt}")
    receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
    expected = receipt.get("sha256") or {}
    actual = {
        "adapter_model": sha256(source_adapter),
        "recurrent_state": sha256(source_state),
    }
    if receipt.get("status") != "complete" or actual != expected:
        raise ValueError(
            f"Checkpoint cannot be mirrored: receipt={receipt} actual={actual}"
        )

    mirror_dir.mkdir(parents=True, exist_ok=True)
    destination = mirror_dir / checkpoint.name
    destination_receipt = destination / "checkpoint_completion.json"
    if destination_receipt.is_file():
        validate_resume_checkpoint(destination / "recurrent_state.pt")
        return destination

    staging = mirror_dir / f".{checkpoint.name}.partial-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(
        checkpoint,
        staging,
        ignore=shutil.ignore_patterns("checkpoint_completion.json", "*.partial"),
    )
    mirrored = {
        "adapter_model": sha256(staging / "adapter/adapter_model.safetensors"),
        "recurrent_state": sha256(staging / "recurrent_state.pt"),
    }
    if mirrored != expected:
        raise IOError(
            f"Mirrored checkpoint hash mismatch: expected={expected} actual={mirrored}"
        )
    if destination.exists():
        shutil.rmtree(destination)
    staging.replace(destination)
    temporary_receipt = destination / "checkpoint_completion.json.partial"
    publish_text(
        temporary_receipt,
        destination_receipt,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    validate_resume_checkpoint(destination / "recurrent_state.pt")
    return destination


def prune_distributed_checkpoints(
    output_dir: Path, *, retention: int, preserve_rounds: tuple[int, ...] = ()
) -> None:
    """Keep only the newest complete distributed checkpoints.

    Pruning happens only after the new receipt has been atomically published.
    Incomplete directories are retained for diagnosis and remain ineligible
    for resume because they do not have a valid completion receipt.
    """
    if retention < 0:
        raise ValueError("checkpoint retention must be non-negative")
    if retention == 0:
        return
    complete: list[tuple[int, Path]] = []
    for checkpoint in output_dir.glob("checkpoint-round-*"):
        try:
            round_index = int(checkpoint.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if (checkpoint / "checkpoint_completion.json").is_file():
            complete.append((round_index, checkpoint))
    complete.sort()
    keep = set(preserve_rounds)
    keep.update(round_index for round_index, _ in complete[-retention:])
    for round_index, checkpoint in complete:
        if round_index not in keep:
            shutil.rmtree(checkpoint)


def gather_distributed_history(
    local_history: list[dict[str, Any]],
    history_base: list[dict[str, Any]],
    context: DistributedContext,
) -> list[dict[str, Any]] | None:
    if not context.enabled:
        return list(history_base) + list(local_history)
    gathered = [None] * context.world_size if context.is_main else None
    dist.gather_object(local_history, gathered, dst=0)
    if not context.is_main:
        return None
    combined = list(history_base)
    for rank_history in gathered or []:
        combined.extend(rank_history or [])
    combined.sort(key=lambda row: int(row["global_position"]))
    positions = [int(row["global_position"]) for row in combined]
    if len(positions) != len(set(positions)):
        raise ValueError("Distributed history contains duplicate global positions")
    return combined


def rollout_state(
    wrapper: RecurrentStudentModel,
    tokenizer,
    update_texts: list[str],
    alias: str,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    state = wrapper.initial_state(1, device=device)
    for update_text in update_texts:
        update = move(
            encode_prompt(tokenizer, alias_text(update_text, alias), max_length),
            device,
        )
        state = wrapper.update_state(state, **update).state
    return state


def rollout_state_pair(
    wrapper: RecurrentStudentModel,
    tokenizer,
    native_update_texts: list[str],
    donor_update_texts: list[str],
    alias: str,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update native and donor states together to halve causal forward calls."""
    if len(native_update_texts) != len(donor_update_texts):
        raise ValueError("Native/donor causal trajectories must have equal length")
    if not native_update_texts:
        raise ValueError("Causal trajectories cannot be empty")
    states = wrapper.initial_state(2, device=device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for native_text, donor_text in zip(
        native_update_texts, donor_update_texts, strict=True
    ):
        encoded = [
            move(
                encode_prompt(tokenizer, alias_text(text, alias), max_length),
                device,
            )
            for text in (native_text, donor_text)
        ]
        max_len = max(int(batch["input_ids"].shape[1]) for batch in encoded)
        ids = torch.cat(
            [
                F.pad(
                    batch["input_ids"],
                    (0, max_len - int(batch["input_ids"].shape[1])),
                    value=int(pad_id),
                )
                for batch in encoded
            ],
            dim=0,
        )
        attention = torch.cat(
            [
                F.pad(
                    batch["attention_mask"],
                    (0, max_len - int(batch["attention_mask"].shape[1])),
                    value=0,
                )
                for batch in encoded
            ],
            dim=0,
        )
        states = wrapper.update_state(states, ids, attention).state
    return states[0:1], states[1:2]


def decode_nll(
    wrapper: RecurrentStudentModel,
    tokenizer,
    state: torch.Tensor,
    decode_text: str,
    target: str,
    alias: str,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Any]:
    batch = move(
        encode_prompt_target(
            tokenizer, alias_text(decode_text, alias), target, max_length
        ),
        device,
    )
    prompt_ends = batch.pop("prompt_ends")
    output = wrapper.decode_from_state(
        state, **batch, prompt_ends=prompt_ends
    )
    return causal_lm_loss(output.logits, batch["labels"]), batch, output


def decode_candidate_losses(
    wrapper: RecurrentStudentModel,
    tokenizer,
    states: list[torch.Tensor],
    decode_text: str,
    targets: list[str],
    alias: str,
    max_length: int,
    device: torch.device,
    preference_reduction: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode several state/target views in one padded forward pass."""
    if len(states) != len(targets) or not states:
        raise ValueError("Candidate state and target batches must be non-empty and aligned")
    batches = [
        move(
            encode_prompt_target(
                tokenizer,
                alias_text(decode_text, alias),
                str(target),
                max_length,
            ),
            device,
        )
        for target in targets
    ]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    combined = pad_supervised_batches(batches, int(pad_id))
    prompt_ends = combined.pop("prompt_ends")
    output = wrapper.decode_from_state(
        torch.cat(states, dim=0), **combined, prompt_ends=prompt_ends
    )
    losses = causal_lm_loss_per_sequence(
        output.logits, combined["labels"], reduction="mean"
    )
    preference_losses = (
        losses
        if preference_reduction == "mean"
        else causal_lm_loss_per_sequence(
            output.logits, combined["labels"], reduction=preference_reduction
        )
    )
    return losses, preference_losses


def causal_pair_objective(
    wrapper: RecurrentStudentModel,
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
    device: torch.device,
    margin: float,
    preference_reduction: str,
    objective_mode: str,
    native_margin_weight: float,
    donor_margin_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    native_state, donor_state = rollout_state_pair(
        wrapper,
        tokenizer,
        row["native_update_texts"],
        row["donor_update_texts"],
        alias,
        max_length,
        device,
    )
    if objective_mode == "specificity_triplet":
        raise ValueError(
            "specificity_triplet must use memory-safe branchwise backward"
        )
    losses, preference_losses = decode_candidate_losses(
        wrapper,
        tokenizer,
        [native_state, native_state, donor_state, donor_state],
        row["decode_text"],
        [
            row["native_target"],
            row["donor_target"],
            row["donor_target"],
            row["native_target"],
        ],
        alias,
        max_length,
        device,
        preference_reduction,
    )
    loss, margin_loss = paired_preference_loss(
        losses[0],
        losses[1],
        losses[2],
        losses[3],
        margin,
        preference_nlls=(
            preference_losses[0],
            preference_losses[1],
            preference_losses[2],
            preference_losses[3],
        ),
        objective_mode=objective_mode,
        native_margin_weight=native_margin_weight,
        donor_margin_weight=donor_margin_weight,
    )
    with torch.no_grad():
        native_advantage = preference_losses[1] - preference_losses[0]
        donor_advantage = preference_losses[3] - preference_losses[2]
        native_float = native_state.float()
        donor_float = donor_state.float()
        state_cosine = F.cosine_similarity(native_float, donor_float, dim=-1).mean()
        state_delta = torch.linalg.vector_norm(
            donor_float - native_float, dim=-1
        ).mean()
        state_scale = 0.5 * (
            torch.linalg.vector_norm(native_float, dim=-1).mean()
            + torch.linalg.vector_norm(donor_float, dim=-1).mean()
        )
        diagnostics = {
            "causal_objective_phase": objective_mode,
            "causal_native_advantage": float(native_advantage),
            "causal_donor_advantage": float(donor_advantage),
            "causal_preference": float(0.5 * (native_advantage + donor_advantage)),
            "causal_state_cosine": float(state_cosine),
            "causal_state_delta_norm": float(state_delta),
            "causal_state_relative_delta": float(state_delta / state_scale.clamp_min(1e-12)),
        }
    return loss, margin_loss, diagnostics


def backward_paired_causal_objective(
    wrapper: RecurrentStudentModel,
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
    device: torch.device,
    margin: float,
    preference_reduction: str,
    objective_mode: str,
    native_margin_weight: float,
    donor_margin_weight: float,
    weight: float,
    loss_scale: float,
) -> tuple[float, float, dict[str, Any]]:
    """Backpropagate an exact paired objective with at most two decode views.

    The causal branch deliberately uses eval mode so LoRA dropout cannot make
    the intervention target stochastic; gradients still flow through every
    trainable LoRA parameter. A detached two-view prepass determines the hinge
    activation. Native and
    donor branches are then recomputed and backpropagated independently.  This
    preserves the four-view scalar/gradient while releasing each decoder graph
    before constructing the next one, which is required for a 32B layerwise
    state on one 80GB A100.
    """
    native_target = str(row["native_target"])
    donor_target = str(row["donor_target"])
    was_training = wrapper.training
    wrapper.eval()
    with torch.no_grad():
        native_probe, donor_probe = rollout_state_pair(
            wrapper,
            tokenizer,
            row["native_update_texts"],
            row["donor_update_texts"],
            alias,
            max_length,
            device,
        )
        native_probe_losses, native_probe_preferences = decode_candidate_losses(
            wrapper,
            tokenizer,
            [native_probe, native_probe],
            row["decode_text"],
            [native_target, donor_target],
            alias,
            max_length,
            device,
            preference_reduction,
        )
        donor_probe_losses, donor_probe_preferences = decode_candidate_losses(
            wrapper,
            tokenizer,
            [donor_probe, donor_probe],
            row["decode_text"],
            [donor_target, native_target],
            alias,
            max_length,
            device,
            preference_reduction,
        )
        native_advantage_value = float(
            native_probe_preferences[1] - native_probe_preferences[0]
        )
        donor_advantage_value = float(
            donor_probe_preferences[1] - donor_probe_preferences[0]
        )
        gold_loss_value = 0.5 * (
            float(native_probe_losses[0]) + float(donor_probe_losses[0])
        )
        native_float = native_probe.float()
        donor_float = donor_probe.float()
        state_cosine = float(
            F.cosine_similarity(native_float, donor_float, dim=-1).mean()
        )
        state_delta = torch.linalg.vector_norm(
            donor_float - native_float, dim=-1
        ).mean()
        state_scale = 0.5 * (
            torch.linalg.vector_norm(native_float, dim=-1).mean()
            + torch.linalg.vector_norm(donor_float, dim=-1).mean()
        )
        state_delta_value = float(state_delta)
        state_relative_delta_value = float(
            state_delta / state_scale.clamp_min(1e-12)
        )
    del (
        native_probe,
        donor_probe,
        native_probe_losses,
        native_probe_preferences,
        donor_probe_losses,
        donor_probe_preferences,
        native_float,
        donor_float,
        state_delta,
        state_scale,
    )

    native_coefficient, native_constant, donor_coefficient, donor_constant = (
        paired_preference_split_coefficients(
            native_advantage_value,
            donor_advantage_value,
            margin,
            objective_mode=objective_mode,
            native_margin_weight=native_margin_weight,
            donor_margin_weight=donor_margin_weight,
        )
    )

    native_state = rollout_state(
        wrapper,
        tokenizer,
        row["native_update_texts"],
        alias,
        max_length,
        device,
    )
    native_losses, native_preferences = decode_candidate_losses(
        wrapper,
        tokenizer,
        [native_state, native_state],
        row["decode_text"],
        [native_target, donor_target],
        alias,
        max_length,
        device,
        preference_reduction,
    )
    native_advantage = native_preferences[1] - native_preferences[0]
    native_loss = (
        0.5 * native_losses[0]
        + native_coefficient * native_advantage
        + native_advantage.new_tensor(native_constant)
    )
    backward_loss_branch(native_loss, weight, loss_scale)
    del (
        native_loss,
        native_advantage,
        native_losses,
        native_preferences,
        native_state,
    )

    donor_state = rollout_state(
        wrapper,
        tokenizer,
        row["donor_update_texts"],
        alias,
        max_length,
        device,
    )
    donor_losses, donor_preferences = decode_candidate_losses(
        wrapper,
        tokenizer,
        [donor_state, donor_state],
        row["decode_text"],
        [donor_target, native_target],
        alias,
        max_length,
        device,
        preference_reduction,
    )
    donor_advantage = donor_preferences[1] - donor_preferences[0]
    donor_loss = (
        0.5 * donor_losses[0]
        + donor_coefficient * donor_advantage
        + donor_advantage.new_tensor(donor_constant)
    )
    backward_loss_branch(donor_loss, weight, loss_scale)
    del (
        donor_loss,
        donor_advantage,
        donor_losses,
        donor_preferences,
        donor_state,
    )
    wrapper.train(was_training)

    if objective_mode == "symmetric":
        margin_loss_value = max(
            0.0,
            float(margin)
            - 0.5 * (native_advantage_value + donor_advantage_value),
        )
    else:
        weight_sum = float(native_margin_weight + donor_margin_weight)
        margin_loss_value = (
            float(native_margin_weight)
            * max(0.0, float(margin) - native_advantage_value)
            + float(donor_margin_weight)
            * max(0.0, float(margin) - donor_advantage_value)
        ) / weight_sum
    total_loss_value = gold_loss_value + margin_loss_value
    diagnostics = {
        "causal_objective_phase": objective_mode,
        "causal_native_advantage": native_advantage_value,
        "causal_donor_advantage": donor_advantage_value,
        "causal_preference": 0.5
        * (native_advantage_value + donor_advantage_value),
        "causal_state_cosine": state_cosine,
        "causal_state_delta_norm": state_delta_value,
        "causal_state_relative_delta": state_relative_delta_value,
        "causal_backward_graph_max_views": 2,
        "causal_dropout_mode": "eval_deterministic",
    }
    return total_loss_value, margin_loss_value, diagnostics


def backward_specificity_triplet_objective(
    wrapper: RecurrentStudentModel,
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
    device: torch.device,
    margin: float,
    preference_reduction: str,
    random_seed: int,
    weight: float,
    loss_scale: float,
) -> tuple[float, float, dict[str, Any]]:
    """Backpropagate the exact triplet as three independent two-view graphs."""
    if margin < 0:
        raise ValueError("Causal margin must be non-negative")
    margin_value = float(margin)
    native_target = str(row["native_target"])
    donor_target = str(row["donor_target"])
    was_training = wrapper.training
    wrapper.eval()
    with torch.no_grad():
        native_probe_state = rollout_state(
            wrapper,
            tokenizer,
            row["native_update_texts"],
            alias,
            max_length,
            device,
        )
        native_probe_losses, native_probe_preferences = decode_candidate_losses(
            wrapper,
            tokenizer,
            [native_probe_state, native_probe_state],
            row["decode_text"],
            [native_target, donor_target],
            alias,
            max_length,
            device,
            preference_reduction,
        )
        donor_probe_state = rollout_state(
            wrapper,
            tokenizer,
            row["donor_update_texts"],
            alias,
            max_length,
            device,
        )
        donor_probe_losses, donor_probe_preferences = decode_candidate_losses(
            wrapper,
            tokenizer,
            [donor_probe_state, donor_probe_state],
            row["decode_text"],
            [donor_target, native_target],
            alias,
            max_length,
            device,
            preference_reduction,
        )
        random_state = norm_matched_random_state(native_probe_state, random_seed)
        random_probe_losses, random_probe_preferences = decode_candidate_losses(
            wrapper,
            tokenizer,
            [random_state, random_state],
            row["decode_text"],
            [native_target, donor_target],
            alias,
            max_length,
            device,
            preference_reduction,
        )
        native_advantage_value = float(
            native_probe_preferences[1] - native_probe_preferences[0]
        )
        donor_advantage_value = float(
            donor_probe_preferences[1] - donor_probe_preferences[0]
        )
        random_advantage_value = float(
            random_probe_preferences[1] - random_probe_preferences[0]
        )
        native_float = native_probe_state.float()
        donor_float = donor_probe_state.float()
        state_cosine = float(
            F.cosine_similarity(native_float, donor_float, dim=-1).mean()
        )
        state_delta = torch.linalg.vector_norm(
            donor_float - native_float, dim=-1
        ).mean()
        state_scale = 0.5 * (
            torch.linalg.vector_norm(native_float, dim=-1).mean()
            + torch.linalg.vector_norm(donor_float, dim=-1).mean()
        )
        state_delta_value = float(state_delta)
        state_relative_delta_value = float(
            state_delta / state_scale.clamp_min(1e-12)
        )
    native_active = float(native_advantage_value < margin_value)
    donor_active = float(donor_advantage_value < margin_value)
    random_active = float(random_advantage_value < margin_value)
    specificity_active = float(
        donor_advantage_value + random_advantage_value < 2.0 * margin_value
    )
    del (
        native_probe_losses,
        native_probe_preferences,
        donor_probe_losses,
        donor_probe_preferences,
        random_probe_losses,
        random_probe_preferences,
        native_probe_state,
        donor_probe_state,
        native_float,
        donor_float,
        state_delta,
        state_scale,
    )

    native_state = rollout_state(
        wrapper,
        tokenizer,
        row["native_update_texts"],
        alias,
        max_length,
        device,
    )
    native_losses, native_preferences = decode_candidate_losses(
        wrapper,
        tokenizer,
        [native_state, native_state],
        row["decode_text"],
        [native_target, donor_target],
        alias,
        max_length,
        device,
        preference_reduction,
    )
    native_advantage = native_preferences[1] - native_preferences[0]
    margin_tensor = native_advantage.new_tensor(margin_value)
    native_margin = native_active * (margin_tensor - native_advantage) / 4.0
    native_loss = native_losses[0] / 3.0 + native_margin
    native_loss_value = float(native_loss.detach())
    backward_loss_branch(native_loss, weight, loss_scale)
    del (
        native_loss,
        native_margin,
        native_losses,
        native_preferences,
        native_advantage,
        native_state,
    )

    donor_state = rollout_state(
        wrapper,
        tokenizer,
        row["donor_update_texts"],
        alias,
        max_length,
        device,
    )
    donor_losses, donor_preferences = decode_candidate_losses(
        wrapper,
        tokenizer,
        [donor_state, donor_state],
        row["decode_text"],
        [donor_target, native_target],
        alias,
        max_length,
        device,
        preference_reduction,
    )
    donor_advantage = donor_preferences[1] - donor_preferences[0]
    margin_tensor = donor_advantage.new_tensor(margin_value)
    donor_margin = (
        (donor_active + specificity_active)
        * (margin_tensor - donor_advantage)
        / 4.0
    )
    donor_loss = donor_losses[0] / 3.0 + donor_margin
    donor_loss_value = float(donor_loss.detach())
    backward_loss_branch(donor_loss, weight, loss_scale)
    del (
        donor_loss,
        donor_margin,
        donor_losses,
        donor_preferences,
        donor_advantage,
        donor_state,
    )

    random_losses, random_preferences = decode_candidate_losses(
        wrapper,
        tokenizer,
        [random_state, random_state],
        row["decode_text"],
        [native_target, donor_target],
        alias,
        max_length,
        device,
        preference_reduction,
    )
    random_native_advantage = random_preferences[1] - random_preferences[0]
    margin_tensor = random_native_advantage.new_tensor(margin_value)
    random_margin = (
        random_active * (margin_tensor - random_native_advantage)
        + specificity_active * (margin_tensor - random_native_advantage)
    ) / 4.0
    random_loss = random_losses[0] / 3.0 + random_margin
    random_loss_value = float(random_loss.detach())
    backward_loss_branch(random_loss, weight, loss_scale)
    del (
        random_loss,
        random_margin,
        random_losses,
        random_preferences,
        random_native_advantage,
        random_state,
    )

    specificity_advantage = donor_advantage_value + random_advantage_value
    full_margin = (
        max(0.0, margin_value - native_advantage_value)
        + max(0.0, margin_value - donor_advantage_value)
        + max(0.0, margin_value - random_advantage_value)
        + max(0.0, 2.0 * margin_value - specificity_advantage)
    ) / 4.0
    diagnostics = {
        "causal_objective_phase": "specificity_triplet",
        "causal_native_advantage": native_advantage_value,
        "causal_donor_advantage": donor_advantage_value,
        "causal_preference": 0.5 * (
            native_advantage_value + donor_advantage_value
        ),
        "causal_random_native_advantage": random_advantage_value,
        "causal_specificity_advantage": specificity_advantage,
        "causal_state_cosine": state_cosine,
        "causal_state_delta_norm": state_delta_value,
        "causal_state_relative_delta": state_relative_delta_value,
        "causal_backward_graph_max_views": 2,
        "causal_dropout_mode": "eval_deterministic",
    }
    wrapper.train(was_training)
    return (
        native_loss_value + donor_loss_value + random_loss_value,
        full_margin,
        diagnostics,
    )


@torch.no_grad()
def evaluate_causal_pairs(
    wrapper: RecurrentStudentModel,
    tokenizer,
    rows: list[dict[str, Any]],
    alias: str,
    max_length: int,
    limit: int,
) -> dict[str, Any]:
    wrapper.eval()
    device = next(wrapper.parameters()).device
    native_correct = donor_correct = clean_n = clean_flip = 0
    evaluated = 0
    for row in rows[:limit]:
        native_state = rollout_state(
            wrapper,
            tokenizer,
            row["native_update_texts"],
            alias,
            max_length,
            device,
        )
        donor_state = rollout_state(
            wrapper,
            tokenizer,
            row["donor_update_texts"],
            alias,
            max_length,
            device,
        )
        scores = {"native": {}, "donor": {}}
        for label_value in row["candidate_labels"]:
            label = str(label_value)
            target = str(row["candidate_targets"][label])
            for state_name, state in (("native", native_state), ("donor", donor_state)):
                batch = move(
                    encode_prompt_target(
                        tokenizer,
                        alias_text(row["decode_text"], alias),
                        target,
                        max_length,
                    ),
                    device,
                )
                prompt_end = batch.pop("prompt_ends")
                output = wrapper.decode_from_state(
                    state, **batch, prompt_ends=prompt_end
                )
                scores[state_name][label] = sequence_logprob(
                    output.logits, batch["input_ids"], int(prompt_end.item())
                )
        native_prediction = max(scores["native"], key=scores["native"].get)
        donor_prediction = max(scores["donor"], key=scores["donor"].get)
        native_target = str(row["native_target_label"])
        donor_target = str(row["donor_target_label"])
        native_correct += int(native_prediction == native_target)
        donor_correct += int(donor_prediction == donor_target)
        if native_prediction == native_target:
            clean_n += 1
            clean_flip += int(donor_prediction == donor_target)
        evaluated += 1
    wrapper.train()
    return {
        "n": evaluated,
        "native_accuracy": native_correct / max(evaluated, 1),
        "donor_accuracy": donor_correct / max(evaluated, 1),
        "clean_n": clean_n,
        "directed_flip_to_A_on_clean": clean_flip / max(clean_n, 1),
    }


def backward_loss_branch(
    loss: torch.Tensor, weight: float, loss_scale: float
) -> None:
    """Backpropagate one weighted branch of an additive objective.

    Calling this once per independent graph is gradient-equivalent to one
    backward call over their weighted sum, while allowing each graph to be
    released before the next memory-intensive forward pass.
    """
    if weight < 0 or loss_scale <= 0:
        raise ValueError("Loss weight must be non-negative and scale positive")
    if weight == 0:
        return
    if not loss.requires_grad:
        raise RuntimeError(
            "Active loss branch is detached; refusing to silently skip a broken graph"
        )
    (loss * float(weight) * float(loss_scale)).backward()


def layerwise_trajectory_loss(
    recurrent_state: torch.Tensor,
    direct_state: torch.Tensor,
    *,
    cosine_weight: float,
    relative_mse_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match a recurrent trajectory to the frozen direct-state trajectory.

    Cosine-only matching admits a collapsed solution in which examples share a
    direction but lose the layerwise magnitudes used by the frozen decoder.
    Relative MSE keeps both direction and scale while normalizing each layer by
    its teacher energy, so early/high-norm layers cannot dominate all 64 blocks.
    """
    if recurrent_state.ndim not in (3, 4) or recurrent_state.shape != direct_state.shape:
        raise ValueError(
            "Layerwise trajectory tensors must be aligned [B,L,D] or [B,L,S,D], got "
            f"{tuple(recurrent_state.shape)} and {tuple(direct_state.shape)}"
        )
    if cosine_weight < 0 or relative_mse_weight < 0:
        raise ValueError("Trajectory component weights must be non-negative")
    if cosine_weight + relative_mse_weight <= 0:
        raise ValueError("At least one trajectory component must be positive")
    recurrent = recurrent_state.float()
    direct = direct_state.detach().float()
    cosine = 1.0 - F.cosine_similarity(recurrent, direct, dim=-1).mean()
    teacher_energy = direct.square().sum(dim=-1).clamp_min(1e-12)
    relative_mse = ((recurrent - direct).square().sum(dim=-1) / teacher_energy).mean()
    total = float(cosine_weight) * cosine + float(relative_mse_weight) * relative_mse
    return total, cosine, relative_mse


def capture_layerwise_teacher_state(
    wrapper: LayerwiseRecurrentStudentModel,
    tokenizer,
    direct_text: str,
    target: str,
    alias: str,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Capture a deterministic frozen one-shot trajectory for distillation."""
    direct = direct_supervised_batch(
        tokenizer, direct_text, target, alias, max_length, device
    )
    direct.pop("prompt_ends")
    direct_token_count = int(direct["input_ids"].shape[1])
    direct.pop("labels")
    direct["attention_mask"] = direct["attention_mask"].to(
        dtype=wrapper.backbone.get_input_embeddings().weight.dtype
    )
    was_training = wrapper.backbone.training
    wrapper.backbone.eval()
    try:
        with torch.no_grad():
            return wrapper.capture_direct_state(**direct).state.detach().clone()
    finally:
        wrapper.backbone.train(was_training)


def backward_causal_trajectory_objective(
    wrapper: LayerwiseRecurrentStudentModel,
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
    device: torch.device,
    *,
    cosine_weight: float,
    relative_mse_weight: float,
    weight: float,
    loss_scale: float,
) -> tuple[float, dict[str, float]]:
    """Distill native and donor recurrent states without training the decoder."""
    losses: list[float] = []
    cosines: list[float] = []
    relative_mses: list[float] = []
    specifications = (
        ("native_update_texts", "native_direct_text", "native_target"),
        ("donor_update_texts", "donor_direct_text", "donor_target"),
    )
    for updates_key, direct_key, target_key in specifications:
        teacher = capture_layerwise_teacher_state(
            wrapper,
            tokenizer,
            str(row[direct_key]),
            str(row[target_key]),
            alias,
            max_length,
            device,
        )
        recurrent = rollout_state(
            wrapper,
            tokenizer,
            list(row[updates_key]),
            alias,
            max_length,
            device,
        )
        if recurrent is None:
            raise ValueError("Causal trajectory cannot be cold-start")
        loss, cosine, relative_mse = layerwise_trajectory_loss(
            recurrent,
            teacher,
            cosine_weight=cosine_weight,
            relative_mse_weight=relative_mse_weight,
        )
        losses.append(float(loss.detach()))
        cosines.append(float(cosine.detach()))
        relative_mses.append(float(relative_mse.detach()))
        # The reported objective is the mean of native and donor branches.
        backward_loss_branch(loss, 0.5 * weight, loss_scale)
        del loss, cosine, relative_mse, recurrent, teacher
    return 0.5 * sum(losses), {
        "causal_trajectory_cosine_loss": 0.5 * sum(cosines),
        "causal_trajectory_relative_mse": 0.5 * sum(relative_mses),
    }


def final_decoder_norm(model: torch.nn.Module) -> torch.nn.Module:
    """Return the final decoder norm without retaining every layer activation."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = getattr(base, "model", None)
    norm = getattr(decoder, "norm", None)
    if not isinstance(norm, torch.nn.Module):
        raise ValueError("Could not locate the backbone final decoder norm")
    return norm


def backward_training_objective(
    wrapper: RecurrentStudentModel,
    tokenizer,
    row: dict[str, Any],
    causal_row: dict[str, Any] | None,
    cache: TeacherCache | None,
    online_kd_cache: TeacherCache | None,
    args,
    device: torch.device,
    state_token_id: int,
    loss_scale: float,
    training_progress: float,
) -> dict[str, Any]:
    """Backpropagate one real-row objective plus its paired causal view.

    Direct replay, recurrent generation/state consistency, and causal pairing
    are independent autograd graphs.  Backward each additive branch before
    constructing the next one so long examples do not retain all four graphs
    at once.  Gradients still accumulate into the same parameters and are
    synchronized/normalized only after the configured accumulation window.
    """
    torch.cuda.reset_peak_memory_stats(device)
    rendered_alias = getattr(
        args,
        "state_token_rendered_alias",
        args.state_token_alias,
    )
    history_dropout = history_dropout_at_progress(args, training_progress)
    if args.history_dropout_curriculum and args.dual_path_curriculum:
        (
            effective_direct_weight,
            effective_generation_weight,
            online_kd_phase_weight,
        ) = dual_path_curriculum_weights(args, training_progress)
        history_dropped = history_dropout == 1.0
    else:
        history_dropped = (
            random.Random(
                f"history-dropout:{args.seed}:{row.get('example_id')}:{training_progress:.12f}"
            ).random()
            < history_dropout
            if args.history_dropout_curriculum
            else True
        )
        online_kd_phase_weight = 0.0
        if args.history_dropout_curriculum:
            effective_direct_weight = 0.0 if history_dropped else 1.0
            effective_generation_weight = 1.0 if history_dropped else 0.0
        else:
            effective_direct_weight = args.direct_replay_weight
            effective_generation_weight = args.generation_weight
    direct = direct_supervised_batch(
        tokenizer,
        row["direct_text"],
        row["target"],
        rendered_alias,
        args.max_length,
        device,
    )
    direct.pop("prompt_ends")
    # Record the actual padded direct sequence length before the batch is
    # consumed.  Layerwise operation offload and the training receipt both use
    # this value; keeping it adjacent to batch construction prevents either
    # path from reading an undefined or stale token count.
    direct_token_count = int(direct["input_ids"].shape[1])
    direct["attention_mask"] = direct["attention_mask"].to(
        dtype=wrapper.backbone.get_input_embeddings().weight.dtype
    )
    was_training = wrapper.backbone.training
    try:
        wrapper.backbone.eval()
        if isinstance(wrapper, LayerwiseRecurrentStudentModel):
            direct_offload_used = (
                wrapper.operation_checkpointing
                and direct_token_count >= args.direct_offload_min_tokens
            )
            wrapper.offload_next_operation = direct_offload_used
            captured = wrapper.capture_direct_state(
                input_ids=direct["input_ids"],
                attention_mask=direct["attention_mask"],
                labels=direct["labels"],
            )
            direct_output = captured.model_output
            teacher_state = captured.state.detach().clone()
        else:
            direct_offload_used = False
            final_hidden: list[torch.Tensor] = []
            capture = final_decoder_norm(wrapper.backbone).register_forward_hook(
                lambda _module, _inputs, output: final_hidden.append(output)
            )
            try:
                direct_output = wrapper.backbone(
                    input_ids=direct["input_ids"],
                    attention_mask=direct["attention_mask"],
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            finally:
                capture.remove()
            if len(final_hidden) != 1:
                raise ValueError(
                    f"Expected one final hidden-state capture, got {len(final_hidden)}"
                )
            matches = torch.nonzero(
                direct["input_ids"][0].eq(state_token_id), as_tuple=False
            ).flatten()
            if matches.numel() != 1:
                raise ValueError(
                    f"Direct supervised row has {matches.numel()} state carriers"
                )
            teacher_state = final_hidden[0][
                :, int(matches.item())
            ].detach().clone()
        online_teacher = None
        if online_kd_phase_weight > 0 and online_kd_cache is None:
            online_teacher = detached_teacher_topk(
                direct_output.logits,
                direct["labels"],
                args.online_kd_topk,
            )
        if effective_direct_weight > 0:
            if not bool(direct_output.logits.requires_grad):
                direct_output = wrapper.backbone(
                    input_ids=direct["input_ids"],
                    attention_mask=direct["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
            direct_loss = causal_lm_loss(direct_output.logits, direct["labels"])
            direct_loss_value = float(direct_loss.detach())
            backward_loss_branch(
                direct_loss, effective_direct_weight, loss_scale
            )
        else:
            direct_loss = teacher_state.new_zeros((), dtype=torch.float32)
            direct_loss_value = 0.0
    finally:
        wrapper.backbone.train(was_training)
    del direct_output, direct_loss, direct

    state = rollout_state(
        wrapper,
        tokenizer,
        row["update_texts"],
        rendered_alias,
        args.max_length,
        device,
    )
    recurrent_state_present = state is not None
    if state is None:
        # A cold-start row has no observed interaction and therefore no
        # recurrent state tensor.  Its natural carrier embedding is still
        # trained by generation and direct replay, but trajectory consistency
        # is undefined and must not fabricate a pseudo-observation.
        consistency_loss = teacher_state.new_zeros((), dtype=torch.float32)
        consistency_cosine_loss = consistency_loss
        consistency_relative_mse = consistency_loss
    else:
        if isinstance(wrapper, LayerwiseRecurrentStudentModel):
            (
                consistency_loss,
                consistency_cosine_loss,
                consistency_relative_mse,
            ) = layerwise_trajectory_loss(
                state,
                teacher_state,
                cosine_weight=args.trajectory_cosine_weight,
                relative_mse_weight=args.trajectory_relative_mse_weight,
            )
        else:
            consistency_loss = 1.0 - F.cosine_similarity(
                state.float(), teacher_state.float(), dim=-1
            ).mean()
            consistency_cosine_loss = consistency_loss
            consistency_relative_mse = consistency_loss.new_zeros(())

    generation_loss = teacher_state.new_zeros((), dtype=torch.float32)
    kd_loss = teacher_state.new_zeros((), dtype=torch.float32)
    online_kd_loss = teacher_state.new_zeros((), dtype=torch.float32)
    decoded = None
    decode = None
    if not args.trajectory_curriculum_only and effective_generation_weight > 0:
        generation_loss, decode, decoded = decode_nll(
            wrapper,
            tokenizer,
            state,
            row["decode_text"],
            row["target"],
            rendered_alias,
            args.max_length,
            device,
        )

    if args.behavior_kd_weight > 0:
        if cache is None:
            raise ValueError("Positive KD weight has no loaded teacher cache")
        cached = cache.load(str(row["example_id"]))
        kd_loss = topk_teacher_cross_entropy(
            decoded.logits,
            decode["labels"],
            torch.tensor(cached["target_token_ids"]),
            cached["teacher_topk_ids"],
            cached["teacher_topk_logprobs"],
            args.behavior_kd_temperature,
        )

    if online_kd_phase_weight > 0:
        if decoded is None or decode is None:
            raise ValueError("Dual-path online KD requires both answer routes")
        if online_kd_cache is not None:
            cached_online = online_kd_cache.load(str(row["example_id"]))
            teacher_targets = torch.tensor(cached_online["target_token_ids"])
            teacher_topk_ids = cached_online["teacher_topk_ids"]
            teacher_topk_logprobs = cached_online["teacher_topk_logprobs"]
        else:
            if online_teacher is None:
                raise ValueError("Online KD has neither live teacher nor cache")
            teacher_targets, teacher_topk_ids, teacher_topk_logprobs = online_teacher
        online_kd_loss = topk_teacher_cross_entropy(
            decoded.logits,
            decode["labels"],
            teacher_targets,
            teacher_topk_ids,
            teacher_topk_logprobs,
            args.online_kd_temperature,
        )

    generation_loss_value = float(generation_loss.detach())
    kd_loss_value = float(kd_loss.detach())
    online_kd_loss_value = float(online_kd_loss.detach())
    consistency_loss_value = float(consistency_loss.detach())
    consistency_cosine_loss_value = float(consistency_cosine_loss.detach())
    consistency_relative_mse_value = float(consistency_relative_mse.detach())
    recurrent_loss = (
        effective_generation_weight * generation_loss
        + args.behavior_kd_weight * kd_loss
        + args.online_kd_weight * online_kd_phase_weight * online_kd_loss
        + args.consistency_weight * consistency_loss
    )
    if recurrent_objective_active(
        recurrent_state_present=recurrent_state_present,
        generation_weight=effective_generation_weight,
        behavior_kd_weight=args.behavior_kd_weight,
        online_kd_weight=args.online_kd_weight,
        online_kd_phase_weight=online_kd_phase_weight,
        consistency_weight=args.consistency_weight,
    ):
        backward_loss_branch(recurrent_loss, 1.0, loss_scale)
    del (
        recurrent_loss,
        generation_loss,
        kd_loss,
        online_kd_loss,
        consistency_loss,
        state,
        teacher_state,
    )
    if decoded is not None:
        del decoded
    if decode is not None:
        del decode

    causal_loss_value = 0.0
    causal_margin_loss_value = 0.0
    causal_trajectory_loss_value = 0.0
    causal_pair_id = None
    causal_diagnostics: dict[str, Any] = {}
    if causal_row is not None:
        causal_pair_id = str(causal_row["pair_id"])
        if args.causal_trajectory_weight > 0:
            if not isinstance(wrapper, LayerwiseRecurrentStudentModel):
                raise ValueError("Causal trajectory distillation requires layerwise state")
            (
                causal_trajectory_loss_value,
                causal_diagnostics,
            ) = backward_causal_trajectory_objective(
                wrapper,
                tokenizer,
                causal_row,
                rendered_alias,
                args.max_length,
                device,
                cosine_weight=args.trajectory_cosine_weight,
                relative_mse_weight=args.trajectory_relative_mse_weight,
                weight=(
                    args.causal_trajectory_weight
                    * args.causal_exposure_multiplier
                ),
                loss_scale=loss_scale,
            )
            causal_diagnostics["causal_objective_phase"] = "trajectory_distillation"
        elif args.causal_weight > 0:
            causal_objective_phase = args.causal_objective_mode
            native_margin_weight = 1.0
            donor_margin_weight = 1.0
            causal_loss_mode = args.causal_objective_mode
            if args.causal_objective_mode == "donor_directional":
                # Keep the mean-token gold CE for both native and donor states,
                # but apply the intervention margin only to donor -> A.
                causal_loss_mode = "directional"
                native_margin_weight = 0.0
                donor_margin_weight = 1.0
            elif args.causal_objective_mode in {
                "bootstrap_then_symmetric",
                "bootstrap_then_directional",
            }:
                if training_progress < args.causal_bootstrap_fraction:
                    causal_loss_mode = "directional"
                    causal_objective_phase = (
                        "directional"
                        if args.causal_objective_mode == "bootstrap_then_symmetric"
                        else "directional_bootstrap"
                    )
                    native_margin_weight = args.causal_bootstrap_native_weight
                    donor_margin_weight = args.causal_bootstrap_donor_weight
                else:
                    if args.causal_objective_mode == "bootstrap_then_symmetric":
                        causal_loss_mode = "symmetric"
                        causal_objective_phase = "symmetric"
                    else:
                        causal_loss_mode = "directional"
                        causal_objective_phase = "directional_balanced"
            effective_causal_weight = (
                args.causal_weight * args.causal_exposure_multiplier
            )
            random_seed = int.from_bytes(
                hashlib.sha256(
                    f"{args.seed}:{causal_pair_id}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            if args.causal_objective_mode == "specificity_triplet":
                (
                    causal_loss_value,
                    causal_margin_loss_value,
                    causal_diagnostics,
                ) = backward_specificity_triplet_objective(
                    wrapper,
                    tokenizer,
                    causal_row,
                    rendered_alias,
                    args.max_length,
                    device,
                    args.causal_margin,
                    args.causal_preference_reduction,
                    random_seed,
                    effective_causal_weight,
                    loss_scale,
                )
            else:
                (
                    causal_loss_value,
                    causal_margin_loss_value,
                    causal_diagnostics,
                ) = backward_paired_causal_objective(
                    wrapper,
                    tokenizer,
                    causal_row,
                    rendered_alias,
                    args.max_length,
                    device,
                    args.causal_margin,
                    args.causal_preference_reduction,
                    causal_loss_mode,
                    native_margin_weight,
                    donor_margin_weight,
                    effective_causal_weight,
                    loss_scale,
                )
                causal_diagnostics["causal_objective_phase"] = causal_objective_phase

    total_value = (
        effective_generation_weight * generation_loss_value
        + args.behavior_kd_weight * kd_loss_value
        + args.online_kd_weight * online_kd_phase_weight * online_kd_loss_value
        + args.consistency_weight * consistency_loss_value
        + effective_direct_weight * direct_loss_value
        + args.causal_weight * causal_loss_value
        + args.causal_trajectory_weight * causal_trajectory_loss_value
    )
    record = {
        "loss": total_value,
        "generation_loss": generation_loss_value,
        "behavior_kd_loss": kd_loss_value,
        "online_kd_loss": online_kd_loss_value,
        "online_kd_phase_weight": online_kd_phase_weight,
        "consistency_loss": consistency_loss_value,
        "consistency_cosine_loss": consistency_cosine_loss_value,
        "consistency_relative_mse": consistency_relative_mse_value,
        "direct_loss": direct_loss_value,
        "recurrent_state_present": recurrent_state_present,
        "history_dropout": history_dropout,
        "history_dropped": history_dropped,
        "effective_generation_weight": effective_generation_weight,
        "effective_direct_weight": effective_direct_weight,
        "direct_tokens": int(direct_token_count),
        "direct_offload_used": direct_offload_used,
        "causal_loss": causal_loss_value,
        "causal_margin_loss": causal_margin_loss_value,
        "causal_trajectory_loss": causal_trajectory_loss_value,
        "causal_effective_weight": (
            max(args.causal_weight, args.causal_trajectory_weight)
            * args.causal_exposure_multiplier
            if causal_row is not None
            else 0.0
        ),
        "causal_pair_id": causal_pair_id,
        **causal_diagnostics,
        "task": str(row["task"]),
        "source": str(row["source"]),
        "example_id": str(row["example_id"]),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device)
        / (1024**3),
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device)
        / (1024**3),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--init-recurrent-state", type=Path)
    parser.add_argument(
        "--state-architecture",
        choices=("scalar", "layerwise"),
        default="scalar",
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--causal-train-jsonl", type=Path, required=True)
    parser.add_argument("--causal-valid-jsonl", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument(
        "--online-kd-cache",
        type=Path,
        help=(
            "Immutable qualified-teacher top-k cache used in place of the live "
            "teacher answer-logit forward during dual-path curriculum."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--state-token-count", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument(
        "--direct-offload-min-tokens",
        type=int,
        default=512,
        help=(
            "CPU-offload tensors saved during checkpoint recomputation for "
            "direct-replay sequences at or above this length; preserves the "
            "same objective while bounding worst-case A100 memory."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--lr-scheduler", choices=("constant", "cosine"), default="constant"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--generation-weight", type=float, default=1.0)
    parser.add_argument("--behavior-kd-weight", type=float, default=0.0)
    parser.add_argument("--behavior-kd-temperature", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--trajectory-cosine-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-relative-mse-weight", type=float, default=0.0)
    parser.add_argument(
        "--trajectory-curriculum-only",
        action="store_true",
        help="Freeze behavior training and optimize only direct/recurrent trajectories.",
    )
    parser.add_argument("--direct-replay-weight", type=float, default=0.25)
    parser.add_argument(
        "--history-dropout-curriculum",
        action="store_true",
        help=(
            "Blend full-history direct replay into strict state-only generation "
            "with dropout levels 0, .25, .5, .75, 1 over five equal phases."
        ),
    )
    parser.add_argument(
        "--dual-path-curriculum",
        action="store_true",
        help=(
            "Evaluate full-history teacher and state-only answer routes on every "
            "mixed-stage row, using history dropout as deterministic loss weights."
        ),
    )
    parser.add_argument("--online-kd-weight", type=float, default=0.0)
    parser.add_argument("--online-kd-temperature", type=float, default=1.0)
    parser.add_argument("--online-kd-topk", type=int, default=256)
    parser.add_argument("--causal-weight", type=float, default=1.0)
    parser.add_argument("--causal-trajectory-weight", type=float, default=0.0)
    parser.add_argument(
        "--normalize-causal-exposure",
        action="store_true",
        help=(
            "Scale each causal exposure by real_rows/causal_exposures so the "
            "epoch objective uses a mean causal loss independent of replay rate."
        ),
    )
    parser.add_argument("--causal-margin", type=float, default=0.5)
    parser.add_argument(
        "--causal-preference-reduction",
        choices=("mean", "sum"),
        default="mean",
        help=(
            "Reduction used only for donor/native preference margins. Gold CE "
            "remains mean-token NLL; 'sum' matches the frozen C4 sequence-logprob score."
        ),
    )
    parser.add_argument(
        "--causal-objective-mode",
        choices=(
            "symmetric",
            "directional",
            "donor_directional",
            "bootstrap_then_symmetric",
            "bootstrap_then_directional",
            "specificity_triplet",
        ),
        default="symmetric",
        help=(
            "Causal preference curriculum. Bootstrap modes start with an "
            "asymmetric directional hinge to break identical-state gradient "
            "cancellation. The symmetric tail applies one hinge after averaging "
            "the two advantages; the directional tail enforces each state-side "
            "margin separately. 'donor_directional' preserves gold CE on both "
            "states while applying the intervention margin only to donor -> A."
            " 'specificity_triplet' additionally trains the exact norm-matched "
            "random-state negative used by the frozen C4 swap-minus-random gate."
        ),
    )
    parser.add_argument("--causal-bootstrap-fraction", type=float, default=0.25)
    parser.add_argument("--causal-bootstrap-native-weight", type=float, default=0.25)
    parser.add_argument("--causal-bootstrap-donor-weight", type=float, default=1.0)
    parser.add_argument(
        "--causal-repeat-fraction",
        type=float,
        default=0.0,
        help="If positive, rehearse causal pairs on this fraction of real-data steps.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--trainable-projections", action="store_true")
    parser.add_argument("--state-projection-rank", type=int, default=64)
    parser.add_argument("--layerwise-transition-rank", type=int, default=0)
    parser.add_argument(
        "--layerwise-operation-checkpointing",
        action="store_true",
        help=(
            "Checkpoint whole layerwise update/direct/decode operations so "
            "temporary state hooks are reinstalled during backward recomputation."
        ),
    )
    parser.add_argument(
        "--layerwise-transition-only",
        action="store_true",
        help=(
            "Freeze the CF1 backbone/decoder and train only identity-initialized "
            "per-layer update maps; direct capture and decode remain native."
        ),
    )
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--causal-eval-limit", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-retention",
        type=int,
        default=0,
        help="Keep this many newest complete distributed checkpoints; 0 keeps all.",
    )
    parser.add_argument(
        "--checkpoint-milestone-rounds",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Completed distributed rounds retained in addition to the newest "
            "--checkpoint-retention checkpoints for curriculum-boundary audits."
        ),
    )
    parser.add_argument(
        "--checkpoint-mirror-dir",
        type=Path,
        help=(
            "Optional shared-storage directory for receipt-last distributed "
            "checkpoint mirrors used by service-interruption recovery."
        ),
    )
    parser.add_argument(
        "--diagnostic-max-micro-steps",
        type=int,
        help="Run only this many leading micro-steps and write a diagnostic receipt.",
    )
    parser.add_argument(
        "--diagnostic-real-index",
        type=int,
        help=(
            "Diagnostic-only override that places one original train-row index "
            "first; requires --diagnostic-max-micro-steps 1 and disables the "
            "unrelated causal branch for an exact per-row memory probe."
        ),
    )
    parser.add_argument(
        "--diagnostic-stop-after-round",
        type=int,
        help="In distributed mode, stop after this global round and emit a diagnostic receipt.",
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--promoted-real-order-manifest",
        type=Path,
        help=(
            "Full one-epoch row order produced by the task-retention promotion "
            "builder. Requires an exact distributed resume checkpoint and "
            "preserves its completed prefix while reordering only unseen rows."
        ),
    )
    parser.add_argument(
        "--diagnostic-warm-start-checkpoint",
        type=Path,
        help=(
            "Diagnostic-only weight/state warm start from a distributed checkpoint. "
            "Resets optimizer/scheduler and permits a new causal recipe; never an exact resume."
        ),
    )
    parser.add_argument(
        "--diagnostic-start-global-position",
        type=int,
        default=0,
        help=(
            "First global real-row position for a diagnostic warm-start segment. "
            "Requires --diagnostic-warm-start-checkpoint and distributed mode."
        ),
    )
    parser.add_argument(
        "--diagnostic-index-manifest",
        type=Path,
        help=(
            "Diagnostic-only list of original train-row indices that replaces "
            "the warm-start segment after the source checkpoint's original data "
            "order has been verified. Resets optimizer/scheduler and is never an "
            "exact resume."
        ),
    )
    parser.add_argument(
        "--diagnostic-allow-causal-data-transition",
        action="store_true",
        help=(
            "Permit an explicit causal-pair dataset change only for a weight-only "
            "diagnostic warm start; model data and row-order checks remain strict."
        ),
    )
    parser.add_argument(
        "--validation-only-from-distributed-checkpoint",
        action="store_true",
        help=(
            "Finalize validation and artifacts from an already complete distributed "
            "checkpoint on one GPU without executing another optimizer step."
        ),
    )
    parser.add_argument(
        "--finalize-complete-distributed-epoch",
        action="store_true",
        help=(
            "With validation-only recovery, require the exact final distributed "
            "round and emit a normal complete receipt after full exposure audits."
        ),
    )
    parser.add_argument("--distributed-mpi", action="store_true")
    parser.add_argument("--distributed-world-size", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29641)
    parser.add_argument(
        "--init-run-dir",
        type=Path,
        help="Read adapter/recurrent_state from the selected M4 pilot run directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.state_token_count <= 0:
        raise ValueError("--state-token-count must be positive")

    weights = (
        args.generation_weight,
        args.behavior_kd_weight,
        args.consistency_weight,
        args.direct_replay_weight,
        args.causal_weight,
        args.causal_trajectory_weight,
        args.online_kd_weight,
    )
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Loss weights must be non-negative and not all zero")
    if args.behavior_kd_weight > 0 and args.teacher_cache is None:
        raise ValueError("Positive KD weight requires --teacher-cache")
    if args.dual_path_curriculum and not args.history_dropout_curriculum:
        raise ValueError("Dual-path curriculum requires history dropout curriculum")
    if args.dual_path_curriculum and args.online_kd_weight <= 0:
        raise ValueError("Dual-path curriculum requires positive online KD weight")
    if args.online_kd_cache is not None and not args.dual_path_curriculum:
        raise ValueError("Online-KD cache requires the dual-path curriculum")
    if args.online_kd_temperature <= 0 or args.online_kd_topk <= 0:
        raise ValueError("Online KD temperature and top-k must be positive")
    if args.resume_from_checkpoint and args.diagnostic_warm_start_checkpoint:
        raise ValueError("Exact resume and diagnostic warm start are mutually exclusive")
    if (
        args.diagnostic_allow_causal_data_transition
        and args.diagnostic_warm_start_checkpoint is None
        and args.resume_from_checkpoint is None
    ):
        raise ValueError("Causal data transition requires a diagnostic warm start or exact recovery")
    if args.promoted_real_order_manifest is not None:
        if args.resume_from_checkpoint is None:
            raise ValueError("Promoted real order requires an exact resume checkpoint")
        if args.diagnostic_warm_start_checkpoint is not None:
            raise ValueError("Promoted real order cannot use a diagnostic warm start")
        if not args.distributed_mpi:
            raise ValueError("Promoted real order requires distributed mode")
        args.promoted_real_order_manifest_sha256 = file_sha256(
            args.promoted_real_order_manifest
        )
    else:
        args.promoted_real_order_manifest_sha256 = None
    diagnostic_recovery = (
        args.resume_from_checkpoint is not None
        and (
            args.diagnostic_start_global_position != 0
            or args.diagnostic_index_manifest is not None
            or args.diagnostic_stop_after_round is not None
        )
    )
    if args.diagnostic_warm_start_checkpoint is None and not diagnostic_recovery:
        if args.diagnostic_start_global_position != 0:
            raise ValueError(
                "Diagnostic start position requires a warm-start or exact-resume checkpoint"
            )
        if args.diagnostic_index_manifest is not None:
            raise ValueError(
                "Diagnostic index manifest requires a warm-start or exact-resume checkpoint"
            )
    else:
        if not args.distributed_mpi:
            raise ValueError("Diagnostic warm start requires distributed mode")
        if args.diagnostic_stop_after_round is None:
            raise ValueError("Diagnostic warm start requires an explicit stop round")
        if args.diagnostic_start_global_position <= 0:
            raise ValueError("Diagnostic warm start requires a positive start position")
    if args.trajectory_cosine_weight < 0 or args.trajectory_relative_mse_weight < 0:
        raise ValueError("Trajectory component weights must be non-negative")
    if args.trajectory_cosine_weight + args.trajectory_relative_mse_weight <= 0:
        raise ValueError("At least one trajectory component must be positive")
    if args.trajectory_curriculum_only:
        if not args.layerwise_transition_only:
            raise ValueError("Trajectory curriculum requires transition-only training")
        behavior_weights = (
            args.generation_weight,
            args.behavior_kd_weight,
            args.direct_replay_weight,
            args.causal_weight,
        )
        if any(weight != 0 for weight in behavior_weights):
            raise ValueError("Trajectory curriculum must freeze all behavior losses")
        if args.consistency_weight <= 0:
            raise ValueError("Trajectory curriculum requires positive consistency weight")
    if args.state_architecture == "scalar" and args.init_recurrent_state is None:
        raise ValueError("Scalar state training requires --init-recurrent-state")
    if args.state_architecture == "layerwise" and args.trainable_projections:
        raise ValueError("Layerwise state does not use scalar state projections")
    if args.state_architecture == "scalar" and (
        args.layerwise_transition_rank != 0
        or args.layerwise_transition_only
        or args.layerwise_operation_checkpointing
    ):
        raise ValueError("Scalar state cannot use layerwise transition options")
    if args.layerwise_transition_rank < 0:
        raise ValueError("Layerwise transition rank must be non-negative")
    if args.layerwise_transition_only and args.layerwise_transition_rank <= 0:
        raise ValueError("Transition-only training requires a positive layerwise rank")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("Gradient accumulation must be positive")
    if args.direct_offload_min_tokens <= 0:
        raise ValueError("Direct offload token threshold must be positive")
    if args.checkpoint_retention < 0:
        raise ValueError("checkpoint retention must be non-negative")
    if any(value <= 0 for value in args.checkpoint_milestone_rounds) or len(
        set(args.checkpoint_milestone_rounds)
    ) != len(args.checkpoint_milestone_rounds):
        raise ValueError("checkpoint milestone rounds must be unique and positive")
    args.checkpoint_milestone_rounds = tuple(args.checkpoint_milestone_rounds)
    if not 0 <= args.causal_bootstrap_fraction <= 1:
        raise ValueError("Causal bootstrap fraction must be in [0, 1]")
    if (
        args.causal_bootstrap_native_weight < 0
        or args.causal_bootstrap_donor_weight < 0
        or (
            args.causal_bootstrap_native_weight
            + args.causal_bootstrap_donor_weight
            <= 0
        )
    ):
        raise ValueError("Causal bootstrap weights must be non-negative and nonzero")
    context = context_from_environment(args.distributed_mpi)
    if args.validation_only_from_distributed_checkpoint:
        if context.enabled:
            raise ValueError("Validation-only checkpoint finalization must be single-GPU")
        if args.resume_from_checkpoint is None:
            raise ValueError("Validation-only finalization requires --resume-from-checkpoint")
        if args.diagnostic_max_micro_steps is not None:
            raise ValueError("Validation-only finalization does not accept micro-step diagnostics")
    if (
        args.finalize_complete_distributed_epoch
        and not args.validation_only_from_distributed_checkpoint
    ):
        raise ValueError(
            "Complete distributed-epoch finalization requires validation-only recovery"
        )
    if context.enabled:
        if context.world_size != args.distributed_world_size:
            raise ValueError(
                f"Expected world_size={args.distributed_world_size}, got {context.world_size}"
            )
        if args.diagnostic_max_micro_steps is not None:
            raise ValueError("Distributed exact-epoch mode does not support diagnostics")
        if args.diagnostic_stop_after_round is not None \
            and args.diagnostic_stop_after_round <= 0:
            raise ValueError("Distributed diagnostic stop round must be positive")
        if args.save_steps % args.gradient_accumulation_steps != 0:
            raise ValueError(
                "Distributed save_steps must align with gradient accumulation"
            )
    if args.init_run_dir is not None:
        run_adapter = args.init_run_dir / "final_adapter"
        run_state = args.init_run_dir / "recurrent_state.pt"
        run_completion = args.init_run_dir / "completion.json"
        if not (run_adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Pilot adapter is missing: {run_adapter}")
        if not run_state.is_file() or not run_completion.is_file():
            raise FileNotFoundError(f"Pilot recurrent artifacts are missing: {args.init_run_dir}")
        pilot_completion = json.loads(run_completion.read_text(encoding="utf-8"))
        if pilot_completion.get("status") not in {"complete", "diagnostic_complete"}:
            raise ValueError(f"Pilot run is incomplete: {pilot_completion}")
        args.init_adapter = str(run_adapter)
        args.init_recurrent_state = run_state
    initialize(context, args.master_port)
    if args.diagnostic_max_micro_steps is not None and args.diagnostic_max_micro_steps <= 0:
        raise ValueError("Diagnostic micro-steps must be positive")
    if (
        not context.enabled
        and args.diagnostic_stop_after_round is not None
        and not args.validation_only_from_distributed_checkpoint
    ):
        raise ValueError("Diagnostic stop round requires distributed mode")
    if not torch.cuda.is_available():
        raise RuntimeError("M4 training requires CUDA")
    device = torch.device(f"cuda:{context.local_rank}")
    seed_process_rngs(args.seed + context.rank, device)

    train_rows = read_jsonl(args.train_jsonl)
    valid_rows = read_jsonl(args.valid_jsonl)
    causal_rows = read_jsonl(args.causal_train_jsonl)
    causal_valid_rows = read_jsonl(args.causal_valid_jsonl)
    real_order, causal_schedule = build_epoch_schedule(
        len(train_rows), len(causal_rows), args.seed
    )
    if args.causal_repeat_fraction > 0:
        causal_schedule = build_repeated_causal_schedule(
            len(train_rows),
            len(causal_rows),
            args.causal_repeat_fraction,
            args.seed + 1009,
        )
    diagnostic_indices: list[int] | None = None
    if args.diagnostic_index_manifest is not None:
        diagnostic_indices = load_diagnostic_index_manifest(
            args.diagnostic_index_manifest,
            train_rows,
            source_train_sha256=file_sha256(args.train_jsonl),
            start_global_position=args.diagnostic_start_global_position,
        )
        args.diagnostic_index_manifest_sha256 = file_sha256(
            args.diagnostic_index_manifest
        )
    if args.diagnostic_real_index is not None:
        if context.enabled:
            raise ValueError("Per-row diagnostic must be single-GPU")
        if args.diagnostic_max_micro_steps != 1:
            raise ValueError("Per-row diagnostic requires exactly one micro-step")
        if not 0 <= args.diagnostic_real_index < len(train_rows):
            raise ValueError("Diagnostic real-row index is out of range")
        real_order = [args.diagnostic_real_index] + [
            index for index in real_order if index != args.diagnostic_real_index
        ]
        causal_schedule = {}
        args.causal_exposure_multiplier = 1.0
    else:
        args.causal_exposure_multiplier = causal_exposure_multiplier(
            len(train_rows), len(causal_schedule), args.normalize_causal_exposure
        )

    cache = None
    if args.teacher_cache is not None and not args.validation_only_from_distributed_checkpoint:
        cache = TeacherCache(
            args.teacher_cache,
            expected_source_sha256=file_sha256(args.train_jsonl),
            verify_shards=(not context.enabled or context.is_main),
        )
        barrier(context)
        missing = [
            str(row["example_id"])
            for row in train_rows
            if str(row["example_id"]) not in cache.by_example
        ]
        if args.behavior_kd_weight > 0 and missing:
            raise ValueError(f"Teacher cache misses {len(missing)} train examples")

    online_kd_cache = None
    args.online_kd_cache_manifest_sha256 = None
    args.online_kd_cache_integrity_sha256 = None
    if args.online_kd_cache is not None and not args.validation_only_from_distributed_checkpoint:
        expected_teacher_sha = str(
            getattr(args, "external_state_teacher_sha256", "")
        )
        expected_view_sha = str(getattr(args, "state8_natural_view_sha256", ""))
        if len(expected_teacher_sha) != 64 or len(expected_view_sha) != 64:
            raise ValueError(
                "Online-KD cache requires bound external teacher and State8 view hashes"
            )
        online_kd_cache = TeacherCache(
            args.online_kd_cache,
            expected_source_sha256=file_sha256(args.train_jsonl),
            verify_shards=(not context.enabled or context.is_main),
            expected_manifest={
                "cache_role": "online_kd",
                "teacher_adapter_sha256": expected_teacher_sha,
                "prompt_field": "direct_text",
                "state_token_alias": args.state_token_alias,
                "state_token_count": args.state_token_count,
                "state_token_rendered_alias": expand_state_token_alias(
                    args.state_token_alias, args.state_token_count
                ),
                "state8_natural_view": True,
                "state8_natural_view_sha256": expected_view_sha,
                "layerwise_online_kd": True,
                "max_length": args.max_length,
                "top_k": args.online_kd_topk,
                "attn_implementation": "sdpa",
                "dtype": "float32",
                "scope": "renormalized_topk_logits_at_supervised_target_positions",
            },
        )
        args.online_kd_cache_manifest_sha256 = file_sha256(
            args.online_kd_cache / "manifest.json"
        )
        args.online_kd_cache_integrity_sha256 = file_sha256(
            args.online_kd_cache / "integrity.json"
        )
        barrier(context)
        missing_online = [
            str(row["example_id"])
            for row in train_rows
            if str(row["example_id"]) not in online_kd_cache.by_example
        ]
        if missing_online:
            raise ValueError(
                f"Online-KD cache misses {len(missing_online)} train examples"
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    alias_ids = tokenizer(args.state_token_alias, add_special_tokens=False)["input_ids"]
    if len(alias_ids) != 1:
        raise ValueError(f"State alias must be one token, got {alias_ids}")
    state_token_id = int(alias_ids[0])
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": context.local_rank} if context.enabled else "auto",
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    # The layerwise wrapper patches and captures the carrier with temporary
    # decoder-block hooks.  Gradient-checkpoint recomputation happens after
    # those hooks have been removed, so either reentrant or non-reentrant
    # checkpointing would silently compute a different backward graph.  Keep
    # checkpointing for the scalar architecture, but fail closed and measure
    # the real no-checkpoint activation footprint for layerwise training.
    use_gradient_checkpointing = args.state_architecture != "layerwise"
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    adapter_source = Path(args.init_adapter)
    if args.resume_from_checkpoint:
        validate_resume_checkpoint(args.resume_from_checkpoint)
        adapter_source = args.resume_from_checkpoint.parent / "adapter"
    elif args.diagnostic_warm_start_checkpoint:
        validate_resume_checkpoint(args.diagnostic_warm_start_checkpoint)
        adapter_source = args.diagnostic_warm_start_checkpoint.parent / "adapter"
    backbone = load_replicated_data_parallel_adapter(base, adapter_source)
    if args.state_architecture == "scalar":
        initial_payload = torch.load(
            args.init_recurrent_state, map_location="cpu", weights_only=False
        )
        source_config = initial_payload.get("config") or {}
        source_trainable = bool(source_config.get("trainable_projections"))
        if source_trainable != bool(args.trainable_projections):
            raise ValueError(
                "M4 projection mode does not match the selected M3 recurrent state"
            )
        wrapper = RecurrentStudentModel(
            backbone,
            state_token_id=state_token_id,
            trainable_projections=args.trainable_projections,
            state_projection_rank=args.state_projection_rank,
        )
        wrapper.state_norm.to(device)
        wrapper.state_in_proj.to(device)
        wrapper.state_out_proj.to(device)
        load_recurrent_state_payload(wrapper, initial_payload)
    else:
        if bool(getattr(backbone, "is_gradient_checkpointing", False)):
            raise RuntimeError(
                "Layerwise training requires gradient checkpointing disabled"
            )
        wrapper = LayerwiseRecurrentStudentModel(
            backbone,
            state_token_id,
            transition_rank=args.layerwise_transition_rank,
            operation_checkpointing=args.layerwise_operation_checkpointing,
            state_token_count=args.state_token_count,
        )
        wrapper.transition_in.to(device)
        wrapper.transition_out.to(device)
        if args.init_recurrent_state is not None:
            initial_payload = torch.load(
                args.init_recurrent_state, map_location="cpu", weights_only=False
            )
            load_recurrent_state_payload(wrapper, initial_payload)
        if args.diagnostic_warm_start_checkpoint is not None:
            warm_payload = torch.load(
                args.diagnostic_warm_start_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            if warm_payload.get("distributed") is not True:
                raise ValueError("Diagnostic warm start requires a distributed checkpoint")
            warm_world_size = int(warm_payload.get("world_size", -1))
            warm_round = int(warm_payload.get("round", -1))
            expected_start = warm_world_size * warm_round
            if expected_start != args.diagnostic_start_global_position:
                raise ValueError(
                    "Diagnostic warm-start boundary mismatch: "
                    f"checkpoint={expected_start} requested={args.diagnostic_start_global_position}"
                )
            warm_config = warm_payload.get("config") or {}
            warm_state_count = int(warm_config.get("state_token_count", 1))
            if warm_state_count != args.state_token_count:
                # The transition maps are slot-shared, so a legacy one-slot
                # checkpoint is the only capacity-changing initialization we
                # support.  Same-capacity warm starts (for example State8 ->
                # State8 diagnosis rescue) preserve the complete state shape.
                if warm_state_count != 1 or args.state_token_count <= 1:
                    raise ValueError(
                        "State-token warm start must preserve capacity or widen a legacy one-slot source"
                    )
            compatibility = {
                "state_architecture": args.state_architecture,
                "state_token_alias": args.state_token_alias,
                "layerwise_transition_rank": args.layerwise_transition_rank,
                "train_sha256": file_sha256(args.train_jsonl),
                "valid_sha256": file_sha256(args.valid_jsonl),
                "seed": args.seed,
                "max_length": args.max_length,
            }
            if not args.diagnostic_allow_causal_data_transition:
                compatibility.update(
                    causal_train_sha256=file_sha256(args.causal_train_jsonl),
                    causal_valid_sha256=file_sha256(args.causal_valid_jsonl),
                )
            drift = {
                key: {"checkpoint": warm_config.get(key), "current": value}
                for key, value in compatibility.items()
                if warm_config.get(key) != value
            }
            if drift:
                raise ValueError(
                    "Diagnostic warm-start compatibility drift: "
                    + json.dumps(drift, sort_keys=True, default=str)
                )
            if list(warm_payload.get("real_order") or []) != real_order:
                raise ValueError("Diagnostic warm-start real-data order drift")
            load_recurrent_state_payload(wrapper, warm_payload)
            if diagnostic_indices is not None:
                segment_start = int(args.diagnostic_start_global_position)
                segment_end = segment_start + len(diagnostic_indices)
                expected_stop = math.ceil(segment_end / context.world_size)
                if segment_start % context.world_size != 0:
                    raise ValueError(
                        "Diagnostic index-manifest start must align to world size"
                    )
                if segment_end % context.world_size != 0:
                    raise ValueError(
                        "Diagnostic index-manifest length must align to world size"
                    )
                if args.diagnostic_stop_after_round != expected_stop:
                    raise ValueError(
                        "Diagnostic index-manifest stop boundary mismatch: "
                        f"expected={expected_stop} requested={args.diagnostic_stop_after_round}"
                    )
                real_order[segment_start:segment_end] = diagnostic_indices
                if real_order[segment_start:segment_end] != diagnostic_indices:
                    raise AssertionError("Diagnostic index-manifest application failed")
        if args.layerwise_transition_only:
            for parameter in wrapper.backbone.parameters():
                parameter.requires_grad_(False)

    parameters = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    distributed_rounds_full = (
        rounds_for_examples(len(train_rows), context.world_size)
        if context.enabled
        else len(train_rows)
    )
    planned_optimizer_steps = math.ceil(
        distributed_rounds_full / args.gradient_accumulation_steps
    )
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=args.lr_scheduler,
        total_steps=planned_optimizer_steps,
        warmup_ratio=args.warmup_ratio,
    )
    optimizer.zero_grad(set_to_none=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.state_token_rendered_alias = expand_state_token_alias(
        args.state_token_alias, args.state_token_count
    )
    config = vars(args).copy()
    config.update(
        {
            "state_token_id": state_token_id,
            "state_token_count": args.state_token_count,
            "state_token_rendered_alias": args.state_token_rendered_alias,
            "real_train_rows": len(train_rows),
            "causal_train_pairs": len(causal_rows),
            "planned_causal_exposures": len(causal_schedule),
            "full_micro_steps": len(train_rows),
            "diagnostic_max_micro_steps": args.diagnostic_max_micro_steps,
            "planned_optimizer_steps": planned_optimizer_steps,
            "train_sha256": file_sha256(args.train_jsonl),
            "valid_sha256": file_sha256(args.valid_jsonl),
            "causal_train_sha256": file_sha256(args.causal_train_jsonl),
            "causal_valid_sha256": file_sha256(args.causal_valid_jsonl),
            "effective_init_adapter": str(adapter_source),
            "decoder_view": "state_plus_current_query_strict_late_fusion",
            "real_data_exposure": "exactly_one_epoch_without_replacement",
            "causal_pair_exposure": (
                "repeated_uniform_rehearsal"
                if args.causal_repeat_fraction > 0
                else "each_pair_exactly_once_uniformly_spread"
            ),
            "causal_branch_dropout_mode": "eval_deterministic",
            "distributed": context.enabled,
            "distributed_world_size": context.world_size,
            "global_real_rows": len(train_rows),
        }
    )
    history: list[dict[str, Any]] = []
    history_base: list[dict[str, Any]] = []
    start_step = 1
    start_round = 0
    if args.diagnostic_warm_start_checkpoint is not None:
        if args.diagnostic_start_global_position % context.world_size != 0:
            raise ValueError("Diagnostic start position must align to the current world size")
        start_round = args.diagnostic_start_global_position // context.world_size
        config["diagnostic_weight_only_warm_start"] = True
        config["diagnostic_optimizer_reset"] = True
        config["diagnostic_warm_start_source_world_size"] = warm_world_size
        config["diagnostic_warm_start_source_round"] = warm_round
    gradient_receipt = None
    resume_payload = None
    if args.resume_from_checkpoint:
        payload = torch.load(
            args.resume_from_checkpoint, map_location="cpu", weights_only=False
        )
        resume_payload = payload
        checkpoint_config = payload.get("config") or {}
        if checkpoint_config.get("diagnostic_weight_only_warm_start") is True:
            for key in (
                "diagnostic_weight_only_warm_start",
                "diagnostic_optimizer_reset",
                "diagnostic_warm_start_source_world_size",
                "diagnostic_warm_start_source_round",
                "diagnostic_segment_rounds",
                "planned_optimizer_steps",
            ):
                if key in checkpoint_config:
                    config[key] = checkpoint_config[key]
        if context.enabled and not args.validation_only_from_distributed_checkpoint:
            validate_distributed_resume_recipe(checkpoint_config, config)
        load_recurrent_state_payload(wrapper, payload)
        if not args.validation_only_from_distributed_checkpoint:
            optimizer.load_state_dict(payload["optimizer"])
            if "scheduler" not in payload:
                raise ValueError("Resume checkpoint lacks scheduler state")
            scheduler.load_state_dict(payload["scheduler"])
        history = list(payload.get("history") or [])
        history_base = list(history)
        real_order = list(payload["real_order"])
        causal_schedule = {
            int(key): int(value) for key, value in payload["causal_schedule"].items()
        }
        if args.promoted_real_order_manifest is not None:
            real_order = load_promoted_real_order_manifest(
                args.promoted_real_order_manifest,
                train_rows,
                source_train_sha256=file_sha256(args.train_jsonl),
                checkpoint_payload=payload,
            )
            config["promoted_real_order_applied"] = True
            promoted_manifest = json.loads(
                args.promoted_real_order_manifest.read_text(encoding="utf-8")
            )
            config["promoted_real_order_source_round"] = int(
                promoted_manifest["source_checkpoint_round"]
            )
        if args.validation_only_from_distributed_checkpoint:
            if payload.get("distributed") is not True:
                raise ValueError("Validation-only finalization requires a distributed checkpoint")
            checkpoint_world_size = int(payload.get("world_size", -1))
            checkpoint_round = int(payload.get("round", -1))
            if checkpoint_world_size <= 1 or checkpoint_round <= 0:
                raise ValueError("Distributed checkpoint has invalid world size or round")
            if args.finalize_complete_distributed_epoch:
                expected_final_round = rounds_for_examples(
                    len(train_rows), checkpoint_world_size
                )
                if checkpoint_round != expected_final_round:
                    raise ValueError(
                        "Complete distributed-epoch finalization requires the exact "
                        f"final round: expected={expected_final_round} "
                        f"checkpoint={checkpoint_round}"
                    )
            if args.diagnostic_stop_after_round is not None \
                    and checkpoint_round != args.diagnostic_stop_after_round:
                raise ValueError(
                    "Validation-only checkpoint round does not match diagnostic stop round"
                )
            # Preserve the training protocol as the authoritative run config;
            # finalization changes no weights and only records its recovery mode.
            config = dict(payload.get("config") or config)
            config["validation_only_from_distributed_checkpoint"] = True
            config["finalize_complete_distributed_epoch"] = bool(
                args.finalize_complete_distributed_epoch
            )
            config["validation_only_checkpoint"] = str(args.resume_from_checkpoint)
            config["distributed"] = True
            config["distributed_world_size"] = checkpoint_world_size
            config["distributed_rounds"] = checkpoint_round
            config["effective_init_adapter"] = str(adapter_source)
            start_round = checkpoint_round
        elif context.enabled:
            if payload.get("distributed") is not True:
                raise ValueError("Distributed resume checkpoint lacks distributed receipt")
            if int(payload.get("world_size", -1)) != context.world_size:
                raise ValueError("Distributed resume world size drift")
            rank_states = payload.get("rank_rng_states") or []
            if len(rank_states) != context.world_size:
                raise ValueError("Distributed resume lacks one RNG state per rank")
            rank_rng = rank_states[context.rank]
            if int(rank_rng.get("rank", -1)) != context.rank:
                raise ValueError("Distributed resume rank RNG mismatch")
            random.setstate(rank_rng["python_random_state"])
            torch.set_rng_state(rank_rng["torch_random_state"])
            torch.cuda.set_rng_state(
                rank_rng["cuda_random_state"], device=context.local_rank
            )
            start_round = int(payload["round"])
            history = []
        else:
            random.setstate(payload["python_random_state"])
            torch.set_rng_state(payload["torch_random_state"])
            torch.cuda.set_rng_state_all(payload["cuda_random_state_all"])
            start_step = int(payload["step"]) + 1
        receipt = args.output_dir / "gradient_audit.json"
        if receipt.is_file():
            gradient_receipt = json.loads(receipt.read_text())

    wrapper.train()
    total_steps = (
        min(len(train_rows), args.diagnostic_max_micro_steps)
        if args.diagnostic_max_micro_steps is not None
        else len(train_rows)
    )
    if args.validation_only_from_distributed_checkpoint:
        if resume_payload is None:
            raise AssertionError("Validation-only finalization lost its checkpoint payload")
        total_rounds = int(resume_payload["round"])
        full_history = list(history)
    elif context.enabled:
        total_rounds = rounds_for_examples(total_steps, context.world_size)
        if args.diagnostic_stop_after_round is not None:
            if args.diagnostic_stop_after_round > total_rounds:
                raise ValueError(
                    "Distributed diagnostic stop round exceeds the full epoch"
                )
            if args.diagnostic_stop_after_round < start_round:
                raise ValueError(
                    "Distributed diagnostic stop round cannot precede the resume round"
                )
            total_rounds = args.diagnostic_stop_after_round
        config["distributed_rounds"] = total_rounds
        segment_rounds = total_rounds - start_round
        logical_diagnostic_warm_start = (
            args.diagnostic_warm_start_checkpoint is not None
            or config.get("diagnostic_weight_only_warm_start") is True
        )
        if logical_diagnostic_warm_start:
            diagnostic_start_round = (
                int(config["diagnostic_start_global_position"]) // context.world_size
            )
            logical_segment_rounds = total_rounds - diagnostic_start_round
            config["diagnostic_segment_rounds"] = logical_segment_rounds
            config["planned_optimizer_steps"] = math.ceil(
                logical_segment_rounds / args.gradient_accumulation_steps
            )
        else:
            config["diagnostic_segment_rounds"] = None
            config["planned_optimizer_steps"] = math.ceil(
                segment_rounds / args.gradient_accumulation_steps
            )
        local_accumulated_examples = 0
        for round_index in range(start_round, total_rounds):
            position = global_position(round_index, context.rank, context.world_size)
            active_ranks = min(
                context.world_size,
                max(0, total_steps - round_index * context.world_size),
            )
            has_example = position < total_steps
            local_record = None
            if has_example:
                real_index = real_order[position]
                row = train_rows[real_index]
                causal_index = causal_schedule.get(position)
                causal_row = (
                    causal_rows[causal_index] if causal_index is not None else None
                )
                curriculum_context = (
                    seeded_history_dropout(args.seed + position)
                    if args.history_dropout_curriculum
                    else nullcontext()
                )
                with curriculum_context:
                    local_record = backward_training_objective(
                        wrapper,
                        tokenizer,
                        row,
                        causal_row,
                        cache,
                        online_kd_cache,
                        args,
                        device,
                        state_token_id,
                        loss_scale=1.0,
                        training_progress=curriculum_progress(
                            position,
                            total_steps,
                            args.diagnostic_start_global_position
                            if args.diagnostic_start_global_position > 0
                            else 0,
                            (
                                args.diagnostic_stop_after_round * context.world_size
                                if args.diagnostic_stop_after_round is not None
                                else None
                            ),
                        ),
                    )
                local_accumulated_examples += 1
                local_record.update(
                    {
                        "step": position + 1,
                        "round": round_index + 1,
                        "optimizer_step": math.ceil(
                            (round_index + 1) / args.gradient_accumulation_steps
                        ),
                        "rank": context.rank,
                        "global_position": position,
                    }
                )
                history.append(local_record)
            if gradient_receipt is None and state_path_gradient_audit_ready(
                local_record, args
            ):
                gradient_receipt = gradient_audit(wrapper)
                if context.is_main:
                    (args.output_dir / "gradient_audit.json").write_text(
                        json.dumps(gradient_receipt, indent=2, sort_keys=True) + "\n"
                    )
            completed_round = round_index + 1
            should_step = (
                completed_round % args.gradient_accumulation_steps == 0
                or completed_round == total_rounds
            )
            if should_step:
                global_examples = all_reduce_int(local_accumulated_examples, context)
                synchronize_gradients(
                    parameters,
                    context,
                    normalization_count=global_examples,
                )
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                local_accumulated_examples = 0
            if local_record is not None and (
                position == 0 or (round_index + 1) % 10 == 0
            ):
                print(json.dumps(local_record), flush=True)
            if (
                completed_round % args.save_steps == 0
                or completed_round == total_rounds
                or completed_round in args.checkpoint_milestone_rounds
            ):
                save_distributed_checkpoint(
                    wrapper,
                    optimizer,
                    scheduler,
                    args.output_dir,
                    completed_round,
                    config,
                    history,
                    history_base,
                    real_order,
                    causal_schedule,
                    context,
                    retention=args.checkpoint_retention,
                    mirror_dir=args.checkpoint_mirror_dir,
                    preserve_rounds=args.checkpoint_milestone_rounds,
                )
        barrier(context)
    else:
        for step in range(start_step, total_steps + 1):
            real_index = real_order[step - 1]
            row = train_rows[real_index]
            causal_index = causal_schedule.get(step - 1)
            causal_row = causal_rows[causal_index] if causal_index is not None else None
            curriculum_context = (
                seeded_history_dropout(args.seed + step - 1)
                if args.history_dropout_curriculum
                else nullcontext()
            )
            with curriculum_context:
                record = backward_training_objective(
                    wrapper,
                    tokenizer,
                    row,
                    causal_row,
                    cache,
                    online_kd_cache,
                    args,
                    device,
                    state_token_id,
                    loss_scale=1.0 / args.gradient_accumulation_steps,
                    training_progress=(step - 1) / max(total_steps - 1, 1),
                )
            if gradient_receipt is None and state_path_gradient_audit_ready(
                record, args
            ):
                gradient_receipt = gradient_audit(wrapper)
                (args.output_dir / "gradient_audit.json").write_text(
                    json.dumps(gradient_receipt, indent=2, sort_keys=True) + "\n"
                )
            if step % args.gradient_accumulation_steps == 0 or step == total_steps:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            record.update(
                {
                    "step": step,
                    "optimizer_step": math.ceil(step / args.gradient_accumulation_steps),
                    "global_position": step - 1,
                }
            )
            history.append(record)
            if step == 1 or step % 10 == 0:
                print(json.dumps(record), flush=True)
            if step % args.save_steps == 0:
                save_checkpoint(
                    wrapper,
                    optimizer,
                    scheduler,
                    args.output_dir,
                    step,
                    config,
                    history,
                    real_order,
                    causal_schedule,
                )

    if (
        not args.validation_only_from_distributed_checkpoint
        and gradient_receipt is None
    ):
        raise RuntimeError(
            "Gradient audit never observed an active state-dependent objective"
        )
    if not args.validation_only_from_distributed_checkpoint:
        full_history = gather_distributed_history(history, history_base, context)
    # Training collectives are finished once every rank has contributed its
    # history.  Tear down NCCL on every rank before rank zero starts the long,
    # single-GPU validation pass.  Keeping worker ranks in a barrier here makes
    # the NCCL watchdog kill an otherwise healthy rank-zero validation when it
    # exceeds the collective timeout (Qwen3-32B validation commonly does).
    if context.enabled:
        close(context)
        if not context.is_main:
            return
    if full_history is None:
        raise AssertionError("Main rank did not receive distributed history")
    history = full_history
    validation = evaluate(
        wrapper,
        tokenizer,
        valid_rows,
        args.state_token_rendered_alias,
        args.max_length,
        args.eval_limit,
    )
    causal_validation = evaluate_causal_pairs(
        wrapper,
        tokenizer,
        causal_valid_rows,
        args.state_token_rendered_alias,
        args.max_length,
        args.causal_eval_limit,
    )
    final_dir = args.output_dir / "final_adapter"
    wrapper.backbone.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    fsync_file(final_dir / "adapter_model.safetensors")
    fsync_directory(final_dir)
    recurrent_state = args.output_dir / "recurrent_state.pt"
    temporary_state = args.output_dir / "recurrent_state.pt.partial"
    torch.save(recurrent_state_payload(wrapper, config), temporary_state)
    publish_file(temporary_state, recurrent_state)
    (args.output_dir / "trainer_history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(summarize_history(history), indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "causal_validation_metrics.json").write_text(
        json.dumps(causal_validation, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n"
    )
    diagnostic = (
        args.diagnostic_max_micro_steps is not None
        or args.diagnostic_stop_after_round is not None
        or args.diagnostic_warm_start_checkpoint is not None
        or (
            args.validation_only_from_distributed_checkpoint
            and not args.finalize_complete_distributed_epoch
        )
    )
    if args.validation_only_from_distributed_checkpoint:
        checkpoint_world_size = int(resume_payload["world_size"])
        completed_micro_steps = min(total_steps, total_rounds * checkpoint_world_size)
        completed_optimizer_steps = math.ceil(
            total_rounds / int(config["gradient_accumulation_steps"])
        )
    elif context.enabled and args.resume_from_checkpoint is not None:
        # An exact distributed resume is one logical training run.  The
        # gathered history contains the checkpoint prefix plus this attempt's
        # suffix, while the optimizer/scheduler also resume at their original
        # global step.  Report cumulative epoch support rather than only the
        # service-attempt segment; diagnostic warm starts remain segment-only.
        completed_micro_steps = len(history)
        if config.get("diagnostic_weight_only_warm_start") is True:
            logical_start_round = (
                int(config["diagnostic_start_global_position"]) // context.world_size
            )
            completed_optimizer_steps = math.ceil(
                (total_rounds - logical_start_round)
                / args.gradient_accumulation_steps
            )
        else:
            completed_optimizer_steps = math.ceil(
                total_rounds / args.gradient_accumulation_steps
            )
    else:
        completed_rounds = total_rounds - start_round if context.enabled else total_steps
        completed_micro_steps = (
            min(
                max(0, total_steps - start_round * context.world_size),
                completed_rounds * context.world_size,
            )
            if context.enabled
            else total_steps
        )
        completed_optimizer_steps = (
            math.ceil(completed_rounds / args.gradient_accumulation_steps)
            if context.enabled
            else math.ceil(total_steps / args.gradient_accumulation_steps)
        )
    history_integrity = audit_history_integrity(
        history,
        expected_start=(
            int(config["diagnostic_start_global_position"])
            if config.get("diagnostic_weight_only_warm_start") is True
            else 0
        ),
    )
    completion = {
        "status": "diagnostic_complete" if diagnostic else "complete",
        "diagnostic": diagnostic,
        "micro_steps": completed_micro_steps,
        "optimizer_steps": completed_optimizer_steps,
        "real_examples_seen": len({row["example_id"] for row in history}),
        "causal_pairs_seen": sum(
            row.get("causal_pair_id") is not None for row in history
        ),
        "unique_causal_pairs_seen": len(
            {
                str(row["causal_pair_id"])
                for row in history
                if row.get("causal_pair_id") is not None
            }
        ),
        **history_integrity,
        "sha256": {
            "adapter_model": sha256(final_dir / "adapter_model.safetensors"),
            "recurrent_state": sha256(recurrent_state),
            "validation_metrics": sha256(args.output_dir / "validation_metrics.json"),
            "causal_validation_metrics": sha256(
                args.output_dir / "causal_validation_metrics.json"
            ),
            "run_config": sha256(args.output_dir / "run_config.json"),
            "gradient_audit": sha256(args.output_dir / "gradient_audit.json"),
        },
    }
    if not diagnostic:
        completion["real_examples_seen_once"] = (
            completion["real_examples_seen"] == len(train_rows)
        )
        completion["causal_exposure_count_matches_plan"] = (
            completion["causal_pairs_seen"] == len(causal_schedule)
        )
        completion["all_causal_pairs_covered"] = (
            completion["unique_causal_pairs_seen"] == len(causal_rows)
        )
        completion["causal_pairs_seen_once"] = (
            completion["causal_exposure_count_matches_plan"]
            and len(causal_schedule) == len(causal_rows)
        )
        completion["exact_history_support"] = (
            completion["history_integrity_pass"] is True
            and completion["history_rows"] == len(train_rows)
            and completion["history_unique_global_positions"] == len(train_rows)
            and completion["history_unique_examples"] == len(train_rows)
        )
        if (
            not completion["real_examples_seen_once"]
            or not completion["causal_exposure_count_matches_plan"]
            or not completion["all_causal_pairs_covered"]
            or not completion["exact_history_support"]
        ):
            raise RuntimeError(f"M4 exposure audit failed: {completion}")
    else:
        completion["contains_causal_pair"] = completion["causal_pairs_seen"] >= 1
        if not completion["contains_causal_pair"]:
            raise RuntimeError(f"M4 diagnostic missed causal path: {completion}")
    temporary_completion = args.output_dir / "completion.json.partial"
    temporary_completion.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    temporary_completion.replace(args.output_dir / "completion.json")
    print(
        json.dumps(
            {"validation": validation, "causal_validation": causal_validation},
            indent=2,
        ),
        flush=True,
    )
    # The distributed process group was deliberately closed before validation;
    # all remaining work is rank-zero-only and must not re-enter a collective.


if __name__ == "__main__":
    main()

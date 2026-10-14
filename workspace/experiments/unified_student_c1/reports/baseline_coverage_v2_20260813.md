# Baseline coverage under public task taxonomy v2

The headline comparison uses public T1--T5. T6 is the appendix-only
future-KC-performance proxy. Internal task keys remain unchanged for artifact
compatibility.

| Public task | Internal key | Completed traditional / task specialist | Same-backbone task-only control | Status and next action |
|---|---|---|---|---|
| T1 correctness | `predict_correctness` | Rasch IRT, PFA, BKT, DKT item-LSTM, official pyKT AKT | **Submitted:** one-epoch Qwen3-32B T1-only, seed 42 | Full 110,525-row task-only train split; downstream full 10,913-row Acc/AUROC evaluator is armed and receipt-bound. |
| T2 concrete response | `predict_response` | Conditional candidate logit + response mode + systematic-error transfer | **Submitted:** one-epoch Qwen3-32B T2-only, seed 42 | Full 55,569-row task-only train split; downstream full 3,693-row MathEq evaluator is armed and receipt-bound. |
| T3 misconception diagnosis | `diagnose_misconception` | BGE-small full 2,586-label taxonomy on Eedi; TF-IDF + LinearSVC on MalRule | **Completed:** full Eedi+MalRule T3-only Qwen; Eedi-only formal specialist remains in progress | Full task-only arm trains on 6,776 rows and reaches 0.546 semantic accuracy on all 1,494 rows with independent TRAPI judgment. Unified reaches 0.566 under the identical judge; paired difference -0.0207, 95% bootstrap CI [-0.0469, 0.0060], McNemar exact p=0.128. |
| T4 prospective behavior | `predict_prospective_error` | PFA-style correctness + conditioned response mode | **Submitted:** one-epoch Qwen3-32B T4-only, seed 42 | Full 78,600-row task-only train split; downstream full 4,600-row JointMath evaluator is armed and receipt-bound. |
| T5 error persistence | `predict_error_persistence` | Persistence logistic survival proxy | **Completed:** one-epoch Qwen3-32B T5-only, seed 42 | Full 8,375-row task-only train split and full 976-row frozen evaluation completed. Accuracy 0.863 versus Unified 0.862; paired difference +0.0010, 95% bootstrap CI [-0.0072, 0.0092], McNemar exact p=1.0. |
| T6 future-KC band (appendix) | `estimate_mastery` | **Completed:** ordinal PFA, class-balanced PFA, same-KC history band, source majority | Not part of the headline task-only matrix | Frozen student-disjoint DBE+XES test has 2,939 rows. Class-balanced PFA reaches 0.439 macro-F1 / 0.487 balanced accuracy / 0.260 QWK; do not call the target latent mastery. |

## Fairness contract for task-only Qwen controls

- Same Qwen3-32B base revision, tokenizer, LoRA rank, max context, decoding, and formal split as the unified checkpoint.
- Select hyperparameters only on the formal validation split; test remains one-shot.
- The one-epoch task-only arms are already exact target-task-exposure controls:
  their unique example IDs and `task`/`target`/`target_label` supervision match
  the corresponding cells of the 259,845-row unified final-fit epoch exactly
  (`results/taskonly_qwen_formal_eval_20260813/target_exposure_audit.json`).
  A total-update-matched arm would have to repeat the same target examples and
  should therefore be labeled a repetition-dose control, not the primary fair
  task-only comparison.
- T3 is open generation with an independent semantic judge; canonical-label exact match is auxiliary.
- T2 and T4 use mathematical equivalence, not raw string exact match, as the primary response metric.
- The Eedi-only T3 arm is not aggregated with the 600-row MalRule test until a model has been trained for both sources.

## Current remote jobs

- State8 interleaved-retention large evaluation: `state8-interleaved-retention-recovery3-20260813`, job `good_juice_nzlc8vp27w`.
- T3-Eedi task-only Qwen formal baseline: `t3-eedi-taskonly-qwen-formal-recovery1-20260813`, job `loyal_chicken_912bh218cq`.
- T1/T2/T4 task-only training: `taskonly-qwen-t1-t2-t4-oneepoch-20260813`.
- T3/T5 task-only training: `taskonly-qwen-t3-t5-oneepoch-20260813`.
- Bound Azure run IDs: T1 `careful_toe_p87sfl2prk`; T2
  `strong_pepper_8ft9yhwng1`; T3 `lemon_button_fyw8ljlmqx`; T4
  `bubbly_carpet_7wbmd8wxx1`; T5 `bright_yam_cpt3f61t59`.
- The latest remote state is intentionally reported as
  `AUTH_OR_STATUS_RETRY` while the project Azure device login is pending; it is
  not interpreted as a training failure or completion.
- Formal evaluation controller: `results/taskonly_qwen_formal_eval_20260813`; it submits each task independently after validating the corresponding remote adapter completion receipt, retains complete per-example predictions, runs T3 through TRAPI, and recomputes the five-cell summary locally.
- T5 same-backbone paired receipt: `results/taskonly_qwen_formal_eval_20260813/paired_interim/T5.json`. The 976 example IDs match exactly; Task-only and Unified are statistically tied (9 Task-only-only correct, 8 Unified-only correct, 833 both correct, 126 both wrong).
- T3 same-backbone paired receipt: `results/taskonly_qwen_formal_eval_20260813/paired_interim/T3.json`. All 1,494 examples were judged with the same TRAPI identity and rubric with zero judge errors. Task-only is 0.546 versus Unified 0.566; the paired interval includes zero, so the formal result is a statistical tie rather than a demonstrated Unified gain.

## Appendix T6 baseline receipt

- Artifact: `results/public_t6_future_kc_baseline_20260814/metrics.json`.
- Test support: 2,939 rows (DBE-KT22 131; XES3G5M 2,808), with zero
  student or example overlap across train/validation/test after source-prefixing.
- The future same-KC outcomes used to construct the label are audited and never
  exposed as model features.
- Ordinary ordinal PFA, selected by validation NLL: accuracy 0.675, macro-F1
  0.353, balanced accuracy 0.373, ordinal MAE 0.399, QWK 0.144.
- Class-balanced ordinal PFA, selected by validation macro-F1: accuracy 0.540,
  macro-F1 0.439, balanced accuracy 0.487, ordinal MAE 0.591, QWK 0.260.
- A 10,000-replicate student-cluster bootstrap stratified by source gives
  class-balanced PFA macro-F1 0.439 [0.420, 0.458], balanced accuracy 0.487
  [0.463, 0.511], and QWK 0.260 [0.226, 0.294]. Its paired balanced-accuracy
  gain over source majority is +0.153 [0.129, 0.178]. Bootstrap receipt:
  `results/public_t6_future_kc_baseline_20260814/bootstrap.json`.
- Source majority reaches accuracy 0.669 but macro-F1 0.267 and QWK 0.000,
  confirming that raw accuracy is dominated by the `high` class and must not be
  the sole appendix metric.

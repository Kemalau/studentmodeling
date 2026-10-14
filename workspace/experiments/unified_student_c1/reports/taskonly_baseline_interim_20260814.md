# Same-backbone task-only baseline interim status

This interim report uses public task taxonomy v2 and the frozen formal test.
All completed comparisons use the same Qwen3-32B base and one-epoch target-task
exposure as the Unified checkpoint. It is superseded only when the full T1--T5
matrix is complete.

| Task | Task-only | Unified | Paired difference (task-only minus Unified) | Status |
|---|---:|---:|---:|---|
| T1 correctness | — | 0.688 Acc | — | One-epoch task-only training running |
| T2 response | — | 0.518 MathEq | — | One-epoch task-only training running |
| T3 diagnosis | 0.546 semantic | 0.566 semantic | -0.0207; 95% bootstrap CI [-0.0469, 0.0060]; McNemar p=0.128 | Complete; statistical tie |
| T4 prospective behavior | — | 0.767 JointMath | — | One-epoch task-only training running |
| T5 persistence | 0.863 Acc | 0.862 Acc | +0.0010; 95% bootstrap CI [-0.0072, 0.0092]; McNemar p=1.000 | Complete; statistical tie |

## Evidence bindings

- T3 task-only generation support: 1,494/1,494; prediction SHA-256
  `24e8a8705b7966fa07c27c8bc0e5b5e936d8ed959a5f4220fc7990dd6afd0719`.
- T3 independent judge: TRAPI `redmond/interactive`, deployment
  `gpt-4o_2024-11-20`, API `2024-10-21`; 1,494 judged and zero errors.
- T3 source cells: Eedi 0.445 (n=894), MalRule cross-template 0.717
  (n=300), MalRule same-template 0.673 (n=300).
- T3 paired receipt:
  `results/taskonly_qwen_formal_eval_20260813/paired_interim/T3.json`.
- T5 paired receipt:
  `results/taskonly_qwen_formal_eval_20260813/paired_interim/T5.json`.
- Exact target-task exposure audit:
  `results/taskonly_qwen_formal_eval_20260813/target_exposure_audit.json`.

The completed T3 and T5 cells do not demonstrate a task-only advantage. They
therefore support the narrower C1 statement that the Unified checkpoint retains
these tasks without a statistically detectable loss relative to same-backbone
single-task training; they do not by themselves prove positive cross-task
transfer (C3).

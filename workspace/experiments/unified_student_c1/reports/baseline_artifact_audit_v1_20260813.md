# Baseline artifact audit v1

This audit separates traditional/source-specialist baselines from the running
same-backbone task-only Qwen controls.  It uses public task taxonomy v2.

| Public task | Completed comparison | Frozen test support | Primary metric | Source artifact |
|---|---|---:|---|---|
| T1 correctness | Rasch IRT, PFA, BKT, DKT item-LSTM, official pyKT AKT | 10,913 (BKT: 9,929 temporal rows) | Accuracy / AUROC | `results/formal_specialist_v1/{irt_pfa,dkt,akt,bkt}.json` |
| T2 response | Conditional candidate logit + response modes + systematic-error transfer | 3,693 | MathEq | `results/formal_specialist_v1/task_specialists.json` |
| T3 diagnosis | BGE-small full taxonomy on Eedi plus TF-IDF+LinearSVC on MalRule | 894 + 600 | Exact class accuracy within each source | `results/formal_specialist_v1/t3_specialists.json` |
| T4 prospective behavior | PFA-style correctness + conditioned response mode | 4,600 | JointMath | `results/formal_specialist_v1/task_specialists.json` |
| T5 persistence | Persistence logistic survival proxy | 976 | Accuracy / AUROC | `results/formal_specialist_v1/task_specialists.json` |

The headline T3 traditional-suite value is an explicitly source-wise weighted
suite, not one universal specialist: `(0.1107382550 × 894 + 0.7016666667 × 600)
/ 1494 = 0.3480589023`.  It must therefore be described as a specialist suite.
The unified and task-only LLM T3 rows instead use open natural-language
generation with an independent semantic judge, so their comparison with exact
taxonomy classifiers is informative but not an identical output protocol.

Artifact SHA-256 bindings:

- `irt_pfa.json`: `e327a773fe93b19065dac49304151913891904fb27643a486079b48254ba78b6`
- `dkt.json`: `1ad5e121bb5e8465ef62de5f9f38a3282ef550778b102b35abbf0196bc493176`
- `akt.json`: `07cb48463d8ea4e9e65819df0b0760c0284f2c8b74d2fc0ad0e8e144f017f48c`
- `bkt.json`: `81c87d8a004a119879c79145e0b859dba5140802284e7981da39198c5b591211`
- `task_specialists.json`: `09b9c3f0ea3b2b0fee183358e14ed2c3dbcea9cbf15ffa3b80ae497853edd6ef`
- `t3_specialists.json`: `ffc9d60cd3a9d0aa4e6dfd35b5554c9c19be0eab9a17028e4b5b736b72be3674`

The same-backbone Qwen controls are intentionally separate: T1/T3/T4/T5 are
running and T2 is queued in Feeds.  Each completed adapter will automatically
receive a full frozen formal evaluation, with T3 judged by TRAPI, before being
inserted into the v3 formal matrix.

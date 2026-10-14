# Selected T1-T6 Dataset Build (2026-07-14)

## Scope

This build uses the five selected sources:

1. DBE-KT22
2. MalruleLib
3. Eedi-Mining
4. Eedi response
5. XES3G5M

FoundationalASSIST is intentionally deferred.

## Task Coverage

| Task | Sources | Operational definition |
|---|---|---|
| T1 `predict_correctness` | DBE-KT22, XES3G5M, Eedi response, MalruleLib | Predict correct vs. incorrect on the query item. |
| T2 `predict_response` | DBE-KT22, Eedi response, MalruleLib | Predict the selected answer, not only correctness. |
| T3 `diagnose_misconception` | Eedi-Mining, MalruleLib | Predict an expert misconception or executable malrule label. |
| T4 `estimate_mastery` | DBE-KT22, XES3G5M | Bin future same-KC correctness into low, medium, or high. The current response is excluded from the future window. |
| T5 `predict_prospective_error` | DBE-KT22 | Show history through position `t`, hide two interactions, and predict correctness plus exact response at `t+3`. |
| T6 `predict_error_persistence` | DBE-KT22, XES3G5M | DBE uses strict same-question/same-wrong-choice recurrence. XES uses same-question incorrect recurrence because selected answers are unavailable. |

## Eedi Response Constraint

The downloaded Eedi response mirror contains `QuestionId`, `UserId`, correctness, correct option, selected option, subjects, and question images, but omits timestamps and `AnswerId`. It is therefore used only for non-temporal user-profile held-out T1/T2. It is not presented as next-step KT, T4, T5, or T6.

All 948 question images were OCRed. OCR marked 782 as structurally usable; the stricter training filter retained 484 questions with meaningful question text, all four answer options, and no indication that the OCR parser moved the question into an option.

## Processed Outputs

| Path | Rows by task |
|---|---|
| `data/processed/dbe_kt22_t1_t6` | T1 4,267; T2 4,267; T4 2,487; T5 3,959; strict T6 147 |
| `data/processed/xes3g5m_t1_t6` | T1 62,148; T4 24,500; response-limited T6 19,037 |
| `data/processed/eedi_response_t1_t2` | T1 17,704; T2 17,704 |
| `data/processed/eedi_response_ocr` | 948 question OCR records |
| `data/processed/mixed_t1_t6_balanced` | 120,000 train; 10,544 test |

The mixed training file has exactly 20,000 rows for each T1-T6 task. Sampling is balanced across available sources where possible, with a maximum repeat count of 8. This prevents the 147 strict DBE T6 rows from being copied dozens of times; XES supplies the remaining T6 quota.

## Length and Leakage Checks

- Tokenizer: local Qwen3-32B tokenizer.
- Maximum encoded length: 1,536 tokens including target and EOS.
- State marker during length validation: existing single-token alias `<|vision_pad|>`.
- Over-length examples are removed before task balancing, so no left truncation removes the student-state path.
- Every query has meaningful question text, and every T4 prompt includes the question text rather than only a KC identifier.
- XES3G5M questions whose stem or options contain image placeholders are excluded because a text-only model cannot recover the missing visual content.
- Student IDs are source-prefixed in the mixed data.
- No train/test student overlap exists, excluding Eedi-Mining's explicit `NO_STUDENT_HISTORY` sentinel.
- Full JSONL validation found no missing fields, duplicate example IDs, invalid targets, or malformed state/query/history sections.

## Main Scripts

- `scripts/student_task_utils.py`
- `scripts/prepare_dbe_kt22_t1_t6.py`
- `scripts/prepare_xes3g5m_t1_t6.py`
- `scripts/ocr_eedi_response_images.py`
- `scripts/prepare_eedi_response_t1_t2.py`
- `scripts/make_mixed_t1_t6_dataset.py`

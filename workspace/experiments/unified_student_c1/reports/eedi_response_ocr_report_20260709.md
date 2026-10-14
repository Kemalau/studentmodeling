# Eedi Response OCR Text Dataset Report

Date: 2026-07-09

## What changed

The previous Eedi response C1 rows used thin pseudo item text such as
`Question 691. Subject: ...` because only `metadata.jsonl` had been downloaded.
The Hugging Face mirror also contains 948 `.jpg` question images. These images
have now been downloaded and converted into text with RapidOCR.

## Outputs

- OCR manifest:
  `experiments/unified_student_c1/data/processed/eedi_response_ocr/questions_ocr.jsonl`
- OCR stats:
  `experiments/unified_student_c1/data/processed/eedi_response_ocr/ocr_stats.json`
- Rebuilt Eedi C1 text data:
  `experiments/unified_student_c1/data/processed/eedi_c1_k12_ocr/`
- Rebuilt mixed no-val C1 data:
  `experiments/unified_student_c1/data/processed/mixed_c1_ocr_no_val/`

## OCR stats

- Total question images: 948
- Parsed as `ok`: 782
- Parsed as `weak`: 166
- Four nonempty options recovered: 657
- Three nonempty options recovered: 130
- Two nonempty options recovered: 102
- Zero nonempty options recovered: 59
- Average OCR confidence: 0.9743

## Rebuilt Eedi C1 stats

- Total examples: 76,581
- Students: 4,918
- Questions used by processed examples: 878
- Skills: 53
- Query OCR status in examples:
  - `ok`: 61,389
  - `weak`: 15,192

Task counts are unchanged from the previous Eedi C1 data:

| Split | next_correctness | next_response | future_mastery |
|---|---:|---:|---:|
| train | 23,604 | 23,604 | 20,853 |
| valid | 1,473 | 1,473 | 1,301 |
| test | 1,479 | 1,479 | 1,315 |

## Schema note

The rebuilt `next_response` target uses the frozen TaskC1 tag:
`<predicted_response>...</predicted_response>`.
The older Eedi rows used `<predicted_answer>...</predicted_answer>`.

## Known limitations

RapidOCR often removes spaces inside English phrases, e.g. `TomandKatie` or
`alwaystrue`. Geometry or image-only option choices are sometimes marked
`weak` because OCR cannot recover the visual answer option. The question-level
manifest keeps raw OCR lines and image paths, so this version can be improved
later with a stronger VLM or manual cleanup without changing the student split.

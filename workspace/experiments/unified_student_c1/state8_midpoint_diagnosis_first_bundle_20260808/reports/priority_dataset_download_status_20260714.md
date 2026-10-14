# Priority Education Dataset Download Status

Date: 2026-07-14

## Summary

| Dataset | Status | Local path | Stored size |
|---|---|---|---:|
| EdNet KT1 + KT3 + contents | Downloaded and CRC-verified | `data/raw/ednet` | 1.9G |
| XES3G5M | Downloaded, gzip-verified and extracted | `data/raw/xes3g5m` | 8.5G |
| DBE-KT22 | Downloaded and inspected | `data/raw/dbe_kt22` | 17M |
| FoundationalASSIST | Blocked by rejected gated-access request | `data/raw/foundationalassist` | README only |

## EdNet

Official source: `https://github.com/riiid/ednet`

Downloaded files:

| File | Compressed bytes | Entries | Uncompressed bytes | SHA-256 |
|---|---:|---:|---:|---|
| `EdNet-KT1.zip` | 1,201,163,816 | 784,310 | 3,072,366,053 | `0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535` |
| `EdNet-KT3.zip` | 762,774,456 | 297,917 | 3,741,834,314 | `cc7823d9a5a0570c81b2c279178547b27ca5c628209f37648f0707484f8efa78` |
| `EdNet-Contents.zip` | 173,976 | 11 | 594,442 | `aa910a0436d9dbac0ba232f55e27b37ad8d39da285fcf15f1c1eb5d064be98e7` |

All three archives passed Python `zipfile.testzip()` CRC validation.

KT1 contains 784,309 student CSV files with:

```text
timestamp, solving_id, question_id, user_answer, elapsed_time
```

KT3 contains 297,915 student CSV files with:

```text
timestamp, action_type, item_id, source, user_answer, platform
```

The content archive was extracted and contains 13,169 questions, 1,021 lectures, payments and coupons.
KT1 and KT3 remain compressed because extracting them would create more than one million small files. A converter
should stream student CSVs directly from each ZIP archive.

## XES3G5M

Official source: `https://github.com/ai4ed/XES3G5M`

- Official Google Drive artifact downloaded as `XES3G5M.tar.gz`.
- Compressed SHA-256: `62d145bd995248f78726a0b6ab69612cf418e3118d0fe01bfe6f4c61b5072f73`.
- The official file was named like a ZIP by the download command but is actually a gzip-compressed tar archive.
- Archive contained 3,886 entries and 8,642,670,992 uncompressed bytes.
- Extraction completed successfully, which also validates the gzip stream CRC.

Important extracted data:

| Resource | Observed size/count |
|---|---:|
| Questions with text/options/answers/analysis | 7,652 |
| KC route map entries | 1,175 |
| Question-level train/validation sequence rows | 30,965 |
| Question-level test student rows | 3,613 |
| KC-level train/validation sequence rows | 33,397 |
| KC-level test student rows | 3,613 |

The question-level sequence data provide question IDs, KC routes, binary responses and timestamps. The metadata
provide Chinese question text, answers, options, worked analysis and images.

## DBE-KT22

Official dataset DOI: `https://doi.org/10.26193/6DZWOH`

The official Dataverse endpoint currently presents a bot-protection challenge to command-line clients. The same raw
tables were downloaded from the public Hugging Face mirror `Unggi/dbe-kt22_raw_data`.

Observed files:

| Table | Rows | Main content |
|---|---:|---|
| `Transaction.csv` | 161,953 | timestamps, answer state/text, selected choice, hint use, selection changes, difficulty/trust feedback |
| `Questions.csv` | 212 | rich text, title, explanation, hints, plain text and difficulty |
| `Question_Choices.csv` | 756 | choice text and correctness |
| `KCs.csv` | 98 | KC names and descriptions |
| `Question_KC_Relationships.csv` | 402 | question-to-KC links |
| `KC_Relationships.csv` | 202 | directed KC relations |
| `Student_Specialization.csv` | 408 | student-specialization links |
| `Specialization.csv` | 8 | specialization names |

`Transaction.csv` SHA-256:
`c5f554ed776bce9ce9a556b21afcab894e963bf8ea5adae94668a98282c13af6`.

The downloaded mirror has 161,953 transaction rows, which is lower than the 167,222 interactions reported in the
paper. This discrepancy must be documented if the dataset is used in experiments.

## FoundationalASSIST

Dataset page: `https://huggingface.co/datasets/ASSISTments/FoundationalASSIST`

- The local Hugging Face account is authenticated.
- Public repository metadata and `README.md` are visible.
- Downloading `Data/Interactions.csv`, `Data/Problems.csv` and `Data/Skills.csv` returns a gated-repository 403.
- Hugging Face reports that this account's access request was rejected by the dataset authors.
- No FoundationalASSIST data rows were downloaded. The local directory contains only the README and cache metadata.

## Next Conversion Work

1. Stream EdNet KT1 from ZIP into student-level canonical events for T1/T2/T4/T5/T6.
2. Convert XES3G5M question-level sequences and join question/KC text for T1/T4 and future correctness.
3. Convert DBE-KT22 transactions into rich text-aware T1/T2/T4/T5 examples.
4. Keep dataset and task sampling balanced; do not concatenate all EdNet events directly with the smaller datasets.
5. Revisit FoundationalASSIST only if the dataset authors approve a new access request.

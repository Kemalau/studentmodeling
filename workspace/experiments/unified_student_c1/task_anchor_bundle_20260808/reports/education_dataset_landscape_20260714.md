# Education Dataset Landscape for Unified Student Modeling

Date: 2026-07-14

This report focuses on datasets relevant to longitudinal student modeling, exact-response prediction,
misconception diagnosis, prospective error prediction, persistence, and tutoring dialogue. It is not an
exhaustive list of every education dataset.

## 1. Recommended Priority

| Priority | Dataset | Main reason | Main limitation |
|---|---|---|---|
| 1 | FoundationalASSIST | Long real sequences, full math text, actual selected and written responses | Manual access approval, non-commercial restrictions |
| 2 | EdNet KT1/KT3 | Very large timestamped sequences with responses and learning actions | TOEIC content is mostly represented by IDs/tags rather than full text |
| 3 | XES3G5M | Long K-12 math sequences plus question text, options, analyses, KCs and images | Chinese data; selected wrong answer is not clearly present in released sequence files |
| 4 | DBE-KT22 | Small enough for rapid experiments but unusually rich metadata and feedback | Narrow university database-exercise domain |
| 5 | Duolingo SLAM | Exact future-error modeling outside mathematics | Token-level language errors rather than math responses |
| 6 | MOOCCubeX | Huge multimodal MOOC behavior and concept graph | Very large and operationally complex |

For the current project, FoundationalASSIST is the strongest candidate for formal T5/T6. EdNet is the
strongest immediately scalable sequence source. XES3G5M is the strongest public text-rich KT alternative.

## 2. Longitudinal Knowledge-Tracing Datasets

### FoundationalASSIST

- Scale: 1,722,169 interactions from 5,000 students over 3,395 math problems and 462 skills.
- Sequence density: about 344 interactions per student on average.
- Content: complete question text, Common Core skill alignment, correctness, actual responses, answer choices,
  and problem type.
- Response richness: approximately 1.2M fill-in responses and 0.5M multiple-choice responses.
- Time span: interactions collected from January 2019 through July 2024.
- Best uses: T1 correctness, T2 exact response, T4 mastery, T5 prospective error, T6 persistence, distractor
  analysis and pedagogical grounding.
- Access: manually gated on Hugging Face; CC BY-NC 4.0 plus responsible-use and secure-storage requirements.
- Fit: the closest existing dataset to the unified generative student-model proposal.

Sources: [dataset card](https://huggingface.co/datasets/ASSISTments/FoundationalASSIST),
[paper](https://arxiv.org/abs/2602.00070).

### EdNet

- Scale: 131,441,538 interactions from 784,309 students collected over two years.
- Content inventory: 13,169 questions, 1,021 lectures and 293 skill types.
- KT1: timestamp, question/bundle, selected option and elapsed time.
- KT2: adds enter/respond/submit actions, intermediate option changes, source and platform.
- KT3: adds explanation reading and lecture watching.
- KT4: adds fine-grained media, choice-elimination, payment and other platform actions.
- Best uses: T1, option-level T2, T4, long-horizon T5/T6, engagement and intervention-effect modeling.
- Limitation: rich behavioral traces but limited natural-language problem content; TOEIC content is largely IDs,
  correct options, parts and tags.
- Access: public downloads, approximately 1.2 GB compressed for KT1 and 0.8 GB for KT3.

Source: [official EdNet repository](https://github.com/riiid/ednet).

### XES3G5M

- Scale: over 5M interactions from more than 18,000 third-grade students; 7,652 valid math questions.
- Sequence fields: question IDs, KC routes, correctness, timestamps and repeated-KC indicators.
- Text metadata: question content, correct answer, options, worked analysis, question type, KC text and images.
- Provides both KC-level and question-level files and five-fold splits.
- Best uses: T1, T4, text-aware KT, multi-step future correctness and representation studies.
- Limitation: the released sequence format emphasizes binary correctness; it does not clearly expose each
  student's selected wrong option, limiting exact T2/T5 error-response construction.
- Access: Google Drive download under the repository's license agreement.

Source: [official XES3G5M repository](https://github.com/ai4ed/XES3G5M).

### DBE-KT22

- Scale: 167,222 interactions, 1,361 students, 212 database exercises and 98 KCs.
- Domain: an Australian National University database course.
- Content: question and KC text, hints, answer choices, images, question difficulty and KC relationships.
- Behavior: answer timing, choice-selection changes and student feedback such as perceived difficulty or
  confidence.
- Best uses: rapid text-aware KT, T1/T2, uncertainty modeling and controlled T5/T6 pilots.
- Limitation: small and domain-specific compared with school-math datasets.
- Access: public through the Australian Data Archive.

Source: [DBE-KT22 paper and access description](https://arxiv.org/abs/2208.12651).

### ASSISTments Legacy Releases

- Releases include 2009-10, 2012-13 with affect, 2014-15/2015 Skill Builder and 2017 variants.
- Common fields include student, problem, skill/KC, opportunity count, correctness, attempts, hints and time.
- Best uses: standard T1/T4 baselines and comparison with BKT, DKT, SAKT and related KT models.
- Limitations: different releases have different schemas and corrected versions; most do not provide complete
  natural-language problem text or exact free responses.
- Access: public dataset pages, but preprocessing choices substantially affect published counts.

Source: [official ASSISTments data site](https://sites.google.com/site/assistmentsdata/home).

### Junyi Academy Math Practicing Log

- Original release: large Taiwanese mathematics-practice logs from 2010-2015.
- Content: student exercise attempts, correctness, time, exercise/KC relations and learning sequences.
- A later 2018-2019 public release is also available through Kaggle.
- Best uses: large-scale T1/T4 and long-horizon sequence experiments.
- Limitations: old release requires substantial filtering, question-KC linkage is incomplete, and access may
  require a DataShop/Kaggle account.

Source: [official DataShop dataset](https://datashop.org/DatasetInfo?datasetId=1198).

### Riiid Answer Correctness Prediction

- Scale: approximately 100M timestamped student-content events from the Riiid education platform.
- Fields: student, timestamp, question/lecture ID, selected answer, correctness, task container, prior response
  time and whether an explanation was viewed.
- Best uses: T1, option-level T2, long-sequence T4/T5/T6 and scalable model benchmarking.
- Limitation: competition data do not provide full licensed TOEIC question text, making semantic LLM claims weak.
- Access: Kaggle account and competition data rules.

Source: [official Kaggle competition](https://www.kaggle.com/competitions/riiid-test-answer-prediction/data).

### KDD Cup 2010 / Cognitive Tutor Algebra

- Algebra tutor step-level logs from the 2005-06 and 2006-07 school years.
- Fields include student, problem hierarchy, problem/step, KC, opportunity, first-attempt correctness, hints,
  attempts and timing.
- Best uses: historical KT baselines, KC modeling and step-level correctness prediction.
- Limitations: old tutor domain, sparse semantic text, known preprocessing and timestamp issues in some subsets.
- Access: free DataShop account required.

Source: [official KDD Cup 2010 DataShop page](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp).

### Eedi Response

- Already downloaded locally: 1,382,727 option responses from 4,918 students on 948 questions.
- Content: student ID, question ID, selected option, correct option, correctness, subject and question image.
- Best uses: T1/T2 and item-prior baselines.
- Current limitation: the public mirror lacks timestamp/AnswerId, so reliable cross-subject longitudinal order
  cannot be reconstructed from this mirror alone.

Source: [public Eedi response mirror](https://huggingface.co/datasets/rik1599/eedi-response-data).

## 3. Misconception and Open-Response Datasets

### Eedi-Mining / Eedi-IR

- 4,370 question-wrong-answer instances linked to 2,587 misconception descriptions.
- Each query contains question text, correct answer, one incorrect option and the gold misconception.
- Best uses: T3 diagnosis, retrieval, and Malrule-to-real-data transfer.
- Limitation: no longitudinal student identity or history.

Source: [Eedi-IR dataset](https://huggingface.co/datasets/cdtmc/eedi-ir).

### MalruleLib

- Executable synthetic mathematical misconceptions with generated problems, correct answers, malrule answers
  and step traces.
- Current local preprocessing uses 21 executable malrules plus `NONE` across subtraction, decimals and fractions.
- Best uses: controlled T2/T3, C4 probing, activation swap and counterfactual experiments.
- Limitation: synthetic behavior cannot replace a real-student headline result.

Source: [official MalruleLib repository](https://github.com/sonkar-lab/malrulelib).

### SciEntsBank

- Short free-text science answers paired with a question and reference answer.
- Labels: correct, contradictory, partially correct/incomplete, irrelevant and non-domain.
- Public mirror contains 4,969 training responses plus unseen-answer, unseen-question and unseen-domain tests.
- Best uses: semantic response assessment and T3-like error-type classification.
- Limitation: no longitudinal student histories.

Source: [public SciEntsBank dataset card](https://huggingface.co/datasets/nkazi/SciEntsBank).

### BEETLE

- Short free-text answers in a basic electricity/electronics tutoring domain.
- Same five-way response-quality labels as SciEntsBank.
- Public mirror contains 3,941 training examples and unseen-answer/unseen-question tests.
- Best uses: response assessment and cross-domain generalization.
- Limitation: no longitudinal student sequence.

Source: [public BEETLE dataset card](https://huggingface.co/datasets/nkazi/Beetle).

## 4. Tutoring Dialogue and Learning-Behavior Datasets

### MathDial

- 2,861 one-to-one teacher-student tutoring dialogues grounded in multi-step math word problems.
- Fields include question, ground-truth answer, incorrect student solution, student profile, teacher-described
  confusion, dialogue acts, full conversation and whether the student eventually self-corrected.
- Best uses: tutor-response generation, misconception-aware feedback and student-simulator evaluation.
- Limitation: student turns were produced by an LLM conditioned on common errors; these are not longitudinal
  real-student records.

Source: [official MathDial repository](https://github.com/eth-nlped/mathdial).

### Duolingo SLAM 2018

- Millions of exercises from thousands of learners over their first 30 days.
- Three tracks: learners of English, Spanish and French.
- Contains anonymized learner IDs, time information, exercise context and token-level mistake annotations.
- Best uses: future exact-error prediction and persistence across languages, closely analogous to T5/T6.
- Limitation: language-token errors rather than mathematics and relatively short 30-day windows.

Source: [official Duolingo SLAM shared task](https://sharedtask.duolingo.com/2018.html).

### OULAD

- More than 30,000 students across seven Open University courses.
- Relational CSV tables cover student demographics, registration, assessments, final outcomes, VLE resources
  and timestamped click behavior.
- Best uses: dropout, engagement, course outcome and coarse mastery prediction.
- Limitation: not designed for exact answer or misconception modeling.

Source: [official UCI OULAD page](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset).

### MOOCCubeX

- 4,216 courses, 230,263 videos, 358,265 exercises, 637,572 concepts and more than 296M behavior events from
  3,330,294 students.
- Resources include video captions, problems, user-problem attempts, video watching, discussions, QA-bot
  interactions and a concept/prerequisite graph.
- Best uses: large-scale learning analytics, recommendation, engagement, concept graphs and multimodal student
  state modeling.
- Limitations: Chinese MOOC domain, very large files, complex relational joins and expensive preprocessing.

Source: [official MOOCCubeX repository](https://github.com/THU-KEG/MOOCCubeX).

## 5. T1-T6 Suitability Matrix

Legend: `strong` means directly supported; `partial` means proxy labels or missing semantic content; `no` means
the released data do not naturally support the task.

| Dataset | T1 correct | T2 response | T3 misconception | T4 mastery | T5 future exact error | T6 persistence |
|---|---|---|---|---|---|---|
| FoundationalASSIST | strong | strong | partial | strong | strong | strong |
| EdNet | strong | option | no | strong | option-level | partial |
| XES3G5M | strong | partial | no | strong | correctness only | partial |
| DBE-KT22 | strong | strong | partial | strong | strong | partial |
| ASSISTments legacy | strong | partial | no | strong | partial | partial |
| Junyi | strong | partial | no | strong | partial | partial |
| Riiid | strong | option | no | strong | option-level | partial |
| Eedi Response mirror | strong | option | no | partial | no reliable order | no reliable order |
| Eedi-Mining | no | wrong option given | strong | no | no | no |
| MalruleLib | strong | strong | strong | no | synthetic | synthetic |
| Duolingo SLAM | token error | strong | linguistic error | partial | strong | strong |
| SciEntsBank/BEETLE | no | open response given | response type | no | no | no |
| MathDial | dialogue outcome | dialogue | free-text confusion | no | simulated | simulated |
| OULAD | assessment outcome | no | no | coarse | no | engagement only |
| MOOCCubeX | partial | partial | no | partial | partial | partial |

## 6. Concrete Acquisition Plan

1. Apply for FoundationalASSIST immediately; it is the best formal C2 dataset.
2. Download EdNet KT1 first and KT3 second; use it to formalize long-horizon sequence construction while the
   gated request is pending.
3. Download DBE-KT22 for fast text-rich pipeline debugging and uncertainty/feedback experiments.
4. Acquire XES3G5M if Chinese math is acceptable; it is the strongest public text-rich KT alternative.
5. Use Duolingo SLAM as a cross-domain demonstration that unified future-error modeling is not math-specific.
6. Keep Eedi-Mining and MalruleLib for T3/C3/C4; they solve a different problem than long-sequence C2 data.

## 7. Bottom Line

No single fully public dataset provides all of long real student histories, complete question text, exact written
responses, misconception labels and persistent-error annotations. FoundationalASSIST comes closest but is gated.
The strongest practical combination is:

- FoundationalASSIST or EdNet for longitudinal T1/T2/T4/T5/T6;
- XES3G5M or DBE-KT22 for text-aware KT;
- Eedi-Mining for real misconception diagnosis;
- MalruleLib for controlled causal-state experiments;
- MathDial for tutoring behavior.

# C1--C4 Claim Taxonomy v2

The T labels enumerate benchmark tasks. The C labels are orthogonal scientific
claims, so one task may provide evidence for more than one claim.

## C1: Unified task coverage

One shared checkpoint and one student-state interface support headline
T1--T5 without task-specific models or catastrophic task collapse. C1 is a
generality claim, not a claim that every task is novel or state of the art.

## C2: Capability frontier beyond correctness-only KT

The model produces educationally meaningful targets that standard
correctness-only knowledge tracing does not natively expose:

- T2 concrete choices or open mathematical responses;
- T3 canonical labels plus natural-language diagnosis and evidence;
- T4 conditional prospective responses and errors;
- T5 strict error-persistence risk.

T3 is a central C2 task. Its authoritative evidence is open generation judged
semantically, not a small oracle candidate list. C1 and C2 overlap: T3 supports
C1 because the unified model covers it, and C2 because its natural-language
diagnosis exceeds the usual correctness-probability interface.

## C3: Cross-task educational transfer

Training on one educational task or supervision type improves a distinct
held-out educational task under exposure-matched controls. Examples include:

- misconception-rule supervision improving real T3 diagnosis;
- response-generation supervision improving T1 or T4;
- T1/T2 supervision improving T5 persistence;
- synthetic rule identity improving authentic expert-labeled diagnosis.

Synthetic-to-real transfer is one important C3 cell, not the definition of C3.
Every positive claim requires matched target-task exposure, total update count,
data source, and seed. A completed negative transfer experiment remains a valid
scientific result and is not converted into a positive claim.

## C4: Reusable student-state validity

The designated state is a useful student variable rather than an arbitrary
hidden activation.  The primary C4 claim is that a frozen, history-closed
State8 representation is predictive, readable, student-specific, and stable:

1. **Predictive sufficiency:** a decoder or frozen readout using only state plus
   the current query retains relevant full-history performance and predicts
   held-out future behavior.
2. **Readable utility:** a readout selected on training/validation students can
   recover future knowledge-component rates and weakness profiles for disjoint
   test students, outperforming both a population prior and a same-window
   correctness/count baseline.
3. **State ownership:** pairing each test student's state with that student's
   own future outcomes performs better than permuting state ownership across
   students.  This is the primary intervention: the readout has no history,
   student-ID, or query channel through which to bypass the state.
4. **Within-student stability:** two disjoint views of one observed history
   produce more consistent downstream predictions than mismatched-student
   views.

The preregistered gates use student-disjoint longitudinal splits, held-out
students, paired bootstrap uncertainty, a state-ownership permutation null,
and a closed information route.  Requiring an arbitrary student's state to
force literal donor diagnosis text is not part of the primary claim: latent
states need not be distributionally valid when transplanted into another
student trajectory.  Cross-student text swaps, same-profile swaps, random
states, and no-op controls remain useful diagnostics and are reported as such,
but they neither establish nor veto C4 by themselves.

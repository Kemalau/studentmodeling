# Pre-inference control-pool correction

The first submission `c4-rule-followup-seed42-v1-20260905` is superseded for an
implementation defect found while all three jobs were last observed queued and
before any model outputs were inspected. The original history pool contains
three task renderings of a shared history. The first builder excluded complete
example IDs only, so selected identity-task instructions for added controls and
allowed some duplicate underlying histories. No scientific interpretation of
that submission is admissible.

Version 2 requires `task == malrule_next_response`, equal response instructions,
and different literal history strings from every original role and the added
same-donor-rule control. The random pool is also deduplicated by history.
All tests explicitly enforce these invariants. Query selection, rule text,
decoding, state intervention, candidate menus, metrics, and the 1,071 endpoint
count remain unchanged. Freeze v2 manifest, code hashes and separate submission.
No outcome selection or hyperparameter tuning informed this correction.

V2 artifacts: `manifests/c4_rule_followup_v2_seed42_20260905.json`,
`manifests/c4_rule_followup_v2_20260905.sha256`,
`amlt/feeds_c4_rule_followup_v2_seed42_20260905.yaml`.
Experiment: `c4-rule-followup-seed42-v2-20260905`.
The original protocol still defines endpoints; this amendment replaces its
additional-control sampling rule and reproduction manifest paths.

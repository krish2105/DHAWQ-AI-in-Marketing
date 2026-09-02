# DHAWQ

## Evaluation

<!-- DHAWQ:EVAL:BEGIN -->
```
DHAWQ — EVALUATION REPORT
Generated 2026-09-03 02:21
Golden set: 60 briefs (v1, draft_v0_unreviewed) · Model: ollama

  ** GOLDEN SET IS UNREVIEWED (draft_v0). Every metric below is PROVISIONAL and must not be reported as measured.

GATES
  ungrounded_claim_rate                0.000    [0.000]   PASS
  citation_validity                    1.000    [1.000]   PASS
  slate_schema_validity                1.000    [1.000]   PASS
  scope_violation_rate                 0.000    [0.000]   PASS
  pii_leak_rate                        0.000    [0.000]   PASS

TUNING
  task_completion_rate                 1.000    [0.850]   PASS
  injection_detection_recall           0.900    [0.900]   PASS
  escalation_precision                 1.000              

OPERATING
  latency_p50_seconds                  0.171              
  latency_p95_seconds                  0.203    [25.000]  PASS
  budget_overrun_rate                  0.000    [0.050]   PASS
  cost_per_brief_usd                   0.000              

INJECTION DETECTION  (split, because the aggregate hides the gap)
  recall_on_designed_payloads          0.900    [0.900]   PASS
  recall_on_novel_payloads             0.000              

BY STRATUM
  standard                    24/24  100.0%
  cold_start                   8/8   100.0%
  constraint_conflicting       8/8   100.0%
  hard_negative                8/8   100.0%
  unanswerable                 6/6   100.0%
  adversarial                  6/6   100.0%

RECOMMENDERS — accuracy vs coverage (the frontier IS the finding)
  model               NDCG@10   MAP@10  coverage    gini    tail  popLift
  popularity           0.0100   0.0042     0.002   0.999   0.000     12.1
  content              0.0060   0.0029     0.624   0.816   0.684      1.4
  collaborative        0.0127   0.0069     0.346   0.865   0.094      4.1
  hybrid_weighted      0.0120   0.0065     0.468   0.840   0.253      3.0
  hybrid_cascade       0.0099   0.0052     0.655   0.764   0.284      3.3

  Cold-start NDCG@10 by training history depth
  model                     0      1-2      3-9      10+
  popularity           0.0181   0.0116   0.0083   0.0071
  content              0.0000   0.0075   0.0060   0.0068
  collaborative        0.0000   0.0168   0.0131   0.0130
  hybrid_weighted      0.0000   0.0149   0.0132   0.0125
  hybrid_cascade       0.0000   0.0075   0.0131   0.0130
```
<!-- DHAWQ:EVAL:END -->

# Revenue Recovery Agent (demo)

Failed-payment dunning and churn triage for a home-services vertical SaaS, executed through the
[verifiable execution harness](../../README.md). The agent proposes, the harness decides, a human
arbitrates the irreversible, and every decision is in a SHA-256 chained ledger.

All data is synthetic (40 accounts, seeded). No Stripe, no CRM, no e-mail is called.

## Run

```bash
cd demos/revenue_recovery
python make_data.py               # generate data/*.csv
python run_demo.py                # queue, agent pass, pending human, ledger, audit
python run_demo.py --approve      # approve pending irreversible actions as gm@demo
python run_demo.py --tamper       # show the audit failing after one value is altered
python -m unittest test_demo      # 8 tests
pip install streamlit && streamlit run app.py   # optional UI
```

Set `ANTHROPIC_API_KEY` (and `pip install anthropic`) to get LLM-personalised drafts; without it,
per-cause templates are used.

## What happens

1. `agent/ingest.py` loads customers, payment attempts, tickets, usage.
2. `agent/classify.py` finds failed payments (cause, consecutive failures) and churn signals
   (ticket open > 7 days, logins down > 50 %). Rules, not ML.
3. `agent/prioritize.py` ranks by dollars at risk = MRR × 12 × rule-based loss probability.
4. `agent/actions.py` proposes one typed action per case, with a reversibility class.
5. `pipeline.py` sends each action through `harness.VerifiableAutonomyLayer.execute()` with the
   4 invariants in `rules.py`. Rejected → rollback, ledger entry, one fallback proposal.
   Irreversible → `PENDING_HUMAN` entry, never executed until a named approver validates.

## Rules (`rules.py`)

| Rule | Enforces |
|---|---|
| `NO_CONTACT_ON_DISPUTE` | no outbound contact while a chargeback is open |
| `MAX_3_CONTACTS_7D` | at most 3 contacts per account in 7 days |
| `AUTO_RETRY_CAP` | auto retry only if amount ≤ 200 $ and cause ∈ {insufficient_funds, card_changed} |
| `NO_DOWNGRADE_WITHOUT_HUMAN` | suspension / downgrade needs a named human approver |

The last rule is enforced twice: by the gate in `pipeline.py` and by the invariant itself, so
bypassing the gate still cannot suspend an account (`test_suspension_without_approver_is_rejected_by_invariant`).

## Scripted scenario (data/)

| Account | Trap | Expected |
|---|---|---|
| C019 | chargeback open | e-mail rejected by `NO_CONTACT_ON_DISPUTE`, held |
| C022 | already contacted 3× | e-mail rejected by `MAX_3_CONTACTS_7D`, held |
| C011 / C028 | 899 $ / 399 $ insufficient funds | retry rejected by `AUTO_RETRY_CAP`, e-mail fallback committed |
| C039 | 3rd consecutive failure | suspension → `PENDING_HUMAN` |
| C003 / C015 / C033 | ≤ 200 $ eligible causes | retry committed |

## Not in the demo

Real payment-processor or CRM integration, trained churn model, multi-tenant, auth. On real data the
first week is read-only: measure the current recovery rate and involuntary churn before any write action.

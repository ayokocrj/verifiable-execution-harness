# -*- coding: utf-8 -*-
"""Streamlit UI for the Revenue Recovery Agent.

    pip install streamlit
    streamlit run app.py
"""
import streamlit as st

from pipeline import RevenueRecoveryRun

st.set_page_config(page_title="Revenue Recovery Agent", layout="wide")

if "run" not in st.session_state:
    rr = RevenueRecoveryRun()
    rr.run_agent()
    st.session_state.run = rr
rr = st.session_state.run

st.title("Revenue Recovery Agent")
st.caption("Failed payments and churn signals, triaged by dollars at risk. The agent proposes, the harness decides, humans arbitrate the irreversible.")

m = rr.metrics()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dollars at risk (annualised)", f"{m['dollars_at_risk']:,.0f} $")
c2.metric("Dollars actioned", f"{m['dollars_actioned']:,.0f} $")
c3.metric("Actions blocked by rules", m["actions_blocked"])
c4.metric("Pending human", m["pending_human"])

tab_queue, tab_pending, tab_ledger, tab_rules = st.tabs(["Queue", "Pending human", "Ledger", "Rules"])

STATUS_ICON = {
    "COMMITTED": "✅", "COMMITTED_AFTER_REJECTION": "✅*", "COMMITTED_BY_HUMAN": "✅👤",
    "PENDING_HUMAN": "⏳", "HELD": "⛔", "REJECTED_BY_HUMAN": "❌👤",
}

with tab_queue:
    rows = []
    for o in rr.outcomes:
        c, a = o["case"], o["action"]
        detail = f"{c['cause']} ×{c['consecutive_failures']} ({c['amount']:.0f} $)" if c["kind"] == "failed_payment" else "; ".join(c["signals"])
        rows.append({
            "": STATUS_ICON[o["status"]], "Account": c["customer_id"], "Company": c["company"], "Kind": c["kind"],
            "Detail": detail, "$ at risk": c["dollars_at_risk"], "Action": a["type"], "Status": o["status"], "Rule": o["rule"] or "",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    pick = st.selectbox("Open a case", [f"{o['case']['customer_id']} · {o['case']['company']}" for o in rr.outcomes])
    o = next(x for x in rr.outcomes if pick.startswith(x["case"]["customer_id"]))
    st.subheader(f"{STATUS_ICON[o['status']]} {o['case']['company']}")
    left, right = st.columns(2)
    with left:
        st.json({k: v for k, v in o["case"].items() if k != "company"})
    with right:
        st.markdown(f"**Proposed action:** `{o['action']['type']}` ({o['action']['reversibility']})")
        if o["rule"]:
            st.warning(f"Rule triggered: **{o['rule']}**")
        body = o["action"]["params"].get("body")
        if body:
            st.text_area("Draft message", body, height=160)

with tab_pending:
    if not rr.pending_human:
        st.info("No irreversible action pending.")
    approver = st.text_input("Your name (recorded in the ledger)", "gm@demo")
    for i, p in enumerate(rr.pending_human):
        c, a = p["case"], p["action"]
        status = next(x["status"] for x in rr.outcomes if x["case"] is c)
        with st.container(border=True):
            st.markdown(f"**{c['company']}** ({c['customer_id']}) · `{a['type']}` · {a['params']['reason']} · MRR {c['mrr']:.0f} $ · status: **{status}**")
            b1, b2, _ = st.columns([1, 1, 4])
            if status == "PENDING_HUMAN":
                if b1.button("Approve", key=f"a{i}"):
                    rr.approve(i, approver)
                    st.rerun()
                if b2.button("Reject", key=f"r{i}"):
                    rr.reject(i, approver)
                    st.rerun()

with tab_ledger:
    ok, msg = rr.layer.ledger.audit()
    (st.success if ok else st.error)(msg)
    rows = []
    for t in rr.layer.ledger.traces:
        act = t.parameters.get("action", {})
        rows.append({
            "Step": t.step, "Hash": t.hash[:16], "Prev": t.previous_hash[:16], "Account": act.get("customer_id", ""),
            "Action": t.action_type, "Rollback": t.rollback_executed, "Rule": t.violation_rule or "",
            "Approver": act.get("params", {}).get("approved_by", t.parameters.get("approver", "")),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if st.button("Tamper with block 1 (demo)"):
        rr.layer.ledger.traces[0].state_after["accounts"]["C025"]["contacts_7d"] = 99
        st.rerun()
    st.download_button("Download ledger.json", rr.layer.ledger.export_json(), "ledger.json")

with tab_rules:
    st.markdown("""
| Rule | What it enforces |
|---|---|
| `NO_CONTACT_ON_DISPUTE` | No outbound contact while a chargeback is open |
| `MAX_3_CONTACTS_7D` | Never more than 3 contacts per account in 7 days |
| `AUTO_RETRY_CAP` | Automatic retry only if amount ≤ 200 $ and cause ∈ {insufficient_funds, card_changed} |
| `NO_DOWNGRADE_WITHOUT_HUMAN` | Suspension or downgrade requires a named human approver |

Reversibility: `send_email`, `schedule_retry` → reversible (autonomous) · `apply_credit` → compensable · `suspend_account` → irreversible (human gate).
""")

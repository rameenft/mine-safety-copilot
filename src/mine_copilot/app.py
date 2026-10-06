"""Streamlit demo: streamlit run src/mine_copilot/app.py

Tests inject `provider` and `tools` through st.session_state; otherwise Gemini + the built DB.
"""

import streamlit as st

from mine_copilot.agent.loop import run
from mine_copilot.agent.tools import YEARS, Tools
from mine_copilot.config import ECFR_DATE
from mine_copilot.demo import (
    AGENT_REPORT,
    EXAMPLES,
    REPO_URL,
    eval_summary,
    friendly_error,
    grounding_problems,
    timed_run,
)
from mine_copilot.llm import GeminiProvider

st.set_page_config(page_title="Mine Safety Copilot", page_icon="⛏️", layout="wide")


@st.cache_resource
def load_tools() -> Tools:
    tools = Tools()
    _ = tools.index  # load chunks + embeddings once, not on the first question
    return tools


def get_tools() -> Tools:
    return st.session_state.get("tools") or load_tools()


def get_provider():
    return st.session_state.get("provider") or GeminiProvider()


# --- sidebar ---------------------------------------------------------------------------------

with st.sidebar:
    st.header("About")
    st.markdown(f"**Model:** `{getattr(get_provider(), 'model', 'fake')}`")
    st.markdown(f"**Regulations:** 30 CFR Part 56 (eCFR snapshot {ECFR_DATE})  \n"
                f"**Accidents:** MSHA records {YEARS[0]}–{YEARS[1]}, surface metal/nonmetal  \n"
                "All fatalities; non-fatal counts come from a 3,000-row sample.")
    if ev := eval_summary():
        st.subheader("Latest eval")
        st.markdown(f"{ev['n']} golden questions on `{ev['model']}`  \n"
                    f"Answer acc **{ev['answer_acc']}** · citation recall **{ev['citation_recall']}**"
                    f"  \nRefusal acc **{ev['refusal_acc']}** · grounded **{ev['grounded']}**  \n"
                    f"[Full report]({REPO_URL}/blob/main/evals/reports/{AGENT_REPORT.name})")
    if meta := st.session_state.get("meta"):
        st.subheader("Last question")
        st.markdown(f"{meta['seconds']}s · {meta['llm_calls']} LLM call{'s' * (meta['llm_calls'] != 1)}  \n"
                    f"{meta['tokens']['input']:,} in / {meta['tokens']['output']:,} out tokens")

# --- question --------------------------------------------------------------------------------

st.title("⛏️ Mine Safety Copilot")
st.caption("Answers only from 30 CFR Part 56 and MSHA accident records, and checks every number "
           "and section it cites against what the tools returned.")

cols = st.columns(len(EXAMPLES))
for col, (label, q) in zip(cols, EXAMPLES.items()):
    if col.button(label, help=q, width="stretch"):
        st.session_state.update(question=q, pending=True)  # example click asks right away

with st.form("ask"):
    question = st.text_input("Ask a safety question", key="question",
                             placeholder="e.g. When are hard hats required?")
    asked = st.form_submit_button("Ask", type="primary")

if (asked or st.session_state.get("pending")) and question.strip():
    st.session_state.pending = False
    with st.spinner("Searching regulations and accident records…"):
        try:
            result, meta = timed_run(run, question.strip(), get_provider(), get_tools())
            st.session_state.update(result=result, meta=meta, error=None)
        except Exception as e:  # noqa: BLE001 — shown to the user, not swallowed
            st.session_state.update(result=None, meta=None, error=friendly_error(e))
    st.rerun()  # so the sidebar picks up this question's tokens/latency

# --- answer ----------------------------------------------------------------------------------

if err := st.session_state.get("error"):
    st.error(err)

if result := st.session_state.get("result"):
    st.subheader(result.question)
    if result.refused:
        st.warning(result.answer, icon="🚫")
    else:
        st.markdown(result.answer)

    if problems := grounding_problems(result):
        st.error("Not fully grounded. These claims don't appear in any tool result: "
                 + ", ".join(problems), icon="⚠️")
    elif not result.refused:
        st.success("Grounded: every number and section cited appears in a tool result.", icon="✅")

    if sections := result.citations.get("sections"):
        st.markdown("**Cited regulations**")
        for sid in sections:
            reg = get_tools().get_regulation(sid)
            with st.expander(f"§ {sid} — {reg.get('heading', '')}"):
                st.markdown(reg.get("text") or reg.get("error", ""))
                if url := reg.get("source_url"):
                    st.markdown(f"[eCFR source]({url})")
    if docs := result.citations.get("documents"):
        st.markdown("**Accident documents cited:** " + ", ".join(f"`{d}`" for d in docs))

    n = len(result.trace)
    with st.expander(f"Tool trace ({n} call{'s' * (n != 1)})"):
        for i, t in enumerate(result.trace, 1):
            st.markdown(f"**{i}. `{t['tool']}`**")
            st.json(t["args"])
            st.json(t["result"], expanded=False)

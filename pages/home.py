"""
Landing page for the combined Knowledge Collapse explorer.
"""
import streamlit as st

st.set_page_config(
    page_title="Knowledge Collapse — Model Explorer",
    page_icon="🧠",
    layout="wide",
)

st.title("Knowledge Collapse — Model Explorer")
st.caption("Acemoglu, Kong & Ozdaglar (2026)  ·  Interactive simulation and extension")

st.markdown("""
> **Core question:** Can AI cause society to *lose* knowledge, even while making individual
> agents more productive on their tasks?
>
> The model says yes — through a public-good externality in learning.
> Choose a model below to explore the mechanism and proposed extensions.
""")

st.divider()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📐  Replication Model")
    st.markdown("""
The original model from Acemoglu, Kong & Ozdaglar (2026).

**What it shows:**
- Agents learn about a shared knowledge state θ_t by exerting costly effort
- Effort produces a *private* signal (for the agent) AND a *public* signal (for the community)
- AI substitutes for effort → agents work less → public knowledge erodes → collapse
- There is a collapse threshold τ_A^c: above it, shared knowledge inevitably falls to zero
- Welfare is hump-shaped in AI capability — some AI is good, too much is catastrophic

**What you can explore:**
- The law of motion F(X_t) and its fixed points
- Time paths from different starting conditions
- Social welfare and the optimal AI policy (garbling)
- How community size protects against collapse (log scaling)
- The public-good externality made visible via agent belief clouds
""")
    if st.button("Open Replication Model →", type="primary", use_container_width=True):
        st.switch_page("knowledge_collapse_ui.py")

with col2:
    st.subheader("🌐  Extension Model")
    st.markdown("""
An extension with cross-disciplinary knowledge recombination.

**What it adds:**
- Three knowledge layers: task (private, resets), domain (persistent, shared), general (sum)
- When AI erodes a domain past a trigger, agents collectively generate a *new* domain
- The new domain inherits knowledge from ALL predecessor domains (convergent inheritance)
- Agent migration is endogenous: credit-equilibrium N_d* ∝ G(X_d)
- Cross-domain complementarity γ: combining knowledge traditions produces synergistic renewal

**Key predictions:**
- **P1:** Open economy (δ > 0) weakens or reverses AI-induced knowledge collapse
- **P2:** The benefit of openness (δ) is *largest* when AI capability is strongest

**What you can explore:**
- Cascading domain dynamics and the staged-generation mechanism
- Open vs closed economy comparison at the same AI level
- Parameter sweeps testing P1 and P2
- The ⚡ strong-reversal config where ΣX_d actually exceeds baseline X_h0
""")
    if st.button("Open Extension Model →", type="primary", use_container_width=True):
        st.switch_page("knowledge_collapse_recomb.py")

st.divider()

with st.expander("ℹ️  How to navigate"):
    st.markdown("""
- Click **Open [Model] →** above to go to that model's interactive explorer.
- Each model has a **🏠 Back to Home** link in its sidebar to return here.
- You can also run each model standalone:
  ```
  streamlit run knowledge_collapse_ui.py       # Replication model only
  streamlit run knowledge_collapse_recomb.py   # Extension model only
  streamlit run app.py                         # Combined explorer (this page)
  ```
- All parameters are adjustable in the sidebar; results update immediately and are cached.
""")

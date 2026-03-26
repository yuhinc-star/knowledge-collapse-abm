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
An extension that adds cross-disciplinary knowledge recombination as a second spillover channel.

**What it adds:**
- Three knowledge layers: task (private, resets), domain (persistent, shared), general (Bayesian sum)
- **Endogenous spawn trigger** (credit-equilibrium): new domain d' created when ΣX_d/k < δ·X_h0 —
  derived from the same G(X) credit function as the effort FOC, not an arbitrary threshold
- **Convergent inheritance**: d' receives δ·(Σ w_d·X_d + γ·Σᵢ<ⱼ wᵢ·Xᵢ·wⱼ·Xⱼ) from ALL predecessor domains
- **Credit-equilibrium migration**: agents allocate as N_d* ∝ G(X_d) — indifferent across domains
- **γ (Weitzman complementarity)**: combining knowledge traditions creates synergy beyond simple pooling

**Five predictions:**
- **P1:** There is a threshold δ* — above it ΣX_d grows, below it decays
- **P2:** ΣX_d trends clearly up or down; almost never flat
- **P3:** AI+recomb always outperforms AI+closed (open economy always helps)
- **P4:** AI+recomb outperforms no-AI+recomb — AI is a *complement* to recombination (activates the cascade)
- **P5:** ΣX_d exceeds pre-AI benchmark X_h0 only when δ > δ*

**What you can explore:**
- Cascading domain dynamics across all spawned domains
- Four-way comparison: closed / open / no-AI / pre-AI benchmark
- P3+P4 (role of AI) and P1+P5 (recombination threshold) sweeps
- The ⚡ strong-reversal config where ΣX_d genuinely exceeds X_h0
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

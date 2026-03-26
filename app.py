"""
Knowledge Collapse — Interactive Model Explorer
Extension of Acemoglu, Kong & Ozdaglar (2026) "AI, Human Cognition and Knowledge Collapse"

Two models:
  Replication  — the original single-domain model: AI substitutes for effort → public signal
                 weakens → shared knowledge X_t collapses. Reproduces all analytical predictions
                 (regime bifurcation, path dependence, non-monotone welfare, logarithmic scaling).

  Extension    — adds cross-disciplinary recombination as a second spillover channel.
                 Multiple domains can spawn endogenously when ΣX_d/k < δ·X_h0 (credit-equilibrium
                 break-even). New domains inherit via convergent weighted inheritance plus Weitzman
                 pairwise synergy (γ). Five predictions tested: P1 threshold δ*, P2 monotone trend,
                 P3 open>closed, P4 AI as complement to recombination, P5 full reversal condition.

Run with:  streamlit run app.py
"""
import streamlit as st

pg = st.navigation(
    [
        st.Page("pages/home.py",                 title="Home",              icon="🏠", default=True),
        st.Page("knowledge_collapse_ui.py",      title="Replication Model", icon="📐"),
        st.Page("knowledge_collapse_recomb.py",  title="Extension Model",   icon="🌐"),
    ],
    position="hidden",   # navigation handled by buttons on the home page and sidebar links
)
pg.run()

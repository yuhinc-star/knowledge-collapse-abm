"""
Combined Knowledge Collapse explorer.

Entry point for the multi-page Streamlit app.
Defines navigation and launches the selected page.

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

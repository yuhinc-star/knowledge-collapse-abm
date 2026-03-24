"""
Interactive UI — Knowledge Collapse ABM
Acemoglu, Kong & Ozdaglar (2026)

Run with:  streamlit run knowledge_collapse_ui.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import streamlit as st
from scipy.stats import norm as _norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

# ── colour palette ────────────────────────────────────────────────────────────
C_HIGH  = "#2563EB"
C_MID   = "#DC2626"
C_ZERO  = "#374151"
C_WELF  = "#059669"
C_AI    = "#D97706"
C_IND   = "#7C3AED"   # individual posterior ellipse
C_COL   = "#0891B2"   # collective posterior ellipse

# ─────────────────────────────────────────────────────────────────────────────
# Core model
# ─────────────────────────────────────────────────────────────────────────────

def G(tau):
    tau = np.asarray(tau, dtype=float)
    return np.where(tau > 0, 2.0 * _norm.cdf(np.sqrt(np.maximum(tau, 0.0))) - 1.0, 0.0)

def g(tau):
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-14)
    return _norm.pdf(np.sqrt(tau)) / np.sqrt(tau)

def _foc(e, X_t, tau_A, alpha, lambda_I, lambda_G, sigma_inv2):
    Y = sigma_inv2 + lambda_I * e + tau_A
    return lambda_I * G(X_t) * g(Y) - e ** (alpha - 1.0)

def solve_effort(X_t, tau_A, alpha, lambda_I, lambda_G, sigma_inv2):
    if X_t <= 0.0:
        return 0.0
    args = (X_t, tau_A, alpha, lambda_I, lambda_G, sigma_inv2)
    if _foc(1e-15, *args) <= 0.0:
        return 0.0
    e_hi = 500.0
    while _foc(e_hi, *args) >= 0.0:
        e_hi *= 10
    return brentq(_foc, 1e-15, e_hi, args=args, xtol=1e-14, maxiter=300)

def F(X_t, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq):
    e     = solve_effort(X_t, tau_A, alpha, lambda_I, lambda_G, sigma_inv2)
    inner = X_t + lambda_G * N * e
    return 1.0 / (1.0 / inner + Sigma_sq)

def find_steady_states(tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq, n_grid=500):
    X_max  = 0.98 / Sigma_sq
    X_log  = np.logspace(-9, np.log10(0.5), n_grid // 2)
    X_lin  = np.linspace(0.5, X_max, n_grid // 2)
    X_grid = np.unique(np.concatenate([X_log, X_lin]))
    kw_ = dict(alpha=alpha, lambda_I=lambda_I, lambda_G=lambda_G,
               sigma_inv2=sigma_inv2, N=N, Sigma_sq=Sigma_sq)
    resid = np.array([F(x, tau_A, **kw_) - x for x in X_grid])
    ss = [0.0]
    for i in range(len(resid) - 1):
        if resid[i] * resid[i + 1] < 0:
            try:
                root = brentq(lambda x: F(x, tau_A, **kw_) - x,
                              X_grid[i], X_grid[i + 1], xtol=1e-8, maxiter=200)
                if not any(abs(root - s) < 1e-4 for s in ss):
                    ss.append(root)
            except Exception:
                pass
    return sorted(ss)

def find_collapse_threshold(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq,
                             tau_lo=0.0, tau_hi=15.0, tol=1e-3):
    kw_ = dict(alpha=alpha, lambda_I=lambda_I, lambda_G=lambda_G,
               sigma_inv2=sigma_inv2, N=N, Sigma_sq=Sigma_sq)
    has_pos = lambda t: any(s > 0.05 for s in find_steady_states(t, **kw_))
    if not has_pos(tau_lo): return tau_lo
    if has_pos(tau_hi):     return tau_hi
    lo, hi = tau_lo, tau_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if has_pos(mid): lo = mid
        else:            hi = mid
    return 0.5 * (lo + hi)

def ss_welfare(tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq):
    kw_ = dict(alpha=alpha, lambda_I=lambda_I, lambda_G=lambda_G,
               sigma_inv2=sigma_inv2, N=N, Sigma_sq=Sigma_sq)
    ss  = find_steady_states(tau_A, **kw_)
    Xh  = max((s for s in ss if s > 0.05), default=0.0)
    if Xh < 1e-6:
        return 0.0, 0.0
    e_bar = solve_effort(Xh, tau_A, alpha, lambda_I, lambda_G, sigma_inv2)
    Y_bar = sigma_inv2 + lambda_I * e_bar + tau_A
    W     = G(Xh) * G(Y_bar) - e_bar ** alpha / alpha
    return float(W), float(Xh)

# ─────────────────────────────────────────────────────────────────────────────
# Agent simulation
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_agents(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq, seed):
    """
    Run N agents for T periods and record per-agent Bayesian belief errors.

    Common state θ_t:
      - Agents inherit collective precision X_t as their prior.
      - Each agent observes their OWN noisy signal with precision λ_G·e
        (individual contribution before aggregation).
      - Individual posterior precision = X_t + λ_G·e  (one agent's view).
      - Collective posterior precision = X_t + λ_G·N·e  (aggregate = public good).
      The gap between individual and collective shows the public-good externality.

    Idiosyncratic state θ_{i,t}:
      - Drawn fresh each period: θ_{i,t} ~ N(0, 1).
      - Agent observes private signal with precision λ_I·e + τ_A.
      - Posterior precision = σ⁻² + λ_I·e + τ_A.
    """
    rng   = np.random.default_rng(seed)
    kw_   = dict(alpha=alpha, lambda_I=lambda_I, lambda_G=lambda_G,
                 sigma_inv2=sigma_inv2, N=N, Sigma_sq=Sigma_sq)

    # Start from high steady state at τ_A=0
    ss0 = find_steady_states(0.0, **kw_)
    X_t = max((s for s in ss0 if s > 0.05), default=1.0)
    theta_t = 0.0

    X_path    = np.zeros(T)
    e_path    = np.zeros(T)
    err_com   = np.zeros((T, N))   # individual belief errors, common state
    err_idi   = np.zeros((T, N))   # individual belief errors, idiosyncratic state
    pvar_ind  = np.zeros(T)        # individual posterior variance (common state)
    pvar_col  = np.zeros(T)        # collective posterior variance (common state)
    pvar_idi  = np.zeros(T)        # posterior variance (idiosyncratic)

    for t in range(T):
        e_bar = solve_effort(X_t, tau_A, alpha, lambda_I, lambda_G, sigma_inv2)

        # ── Common state signals ──────────────────────────────────────────────
        prec_one = lambda_G * e_bar          # one agent's signal precision
        prec_agg = lambda_G * N * e_bar      # aggregate signal precision

        if prec_one > 1e-12:
            z_com = theta_t + rng.normal(0.0, 1.0 / np.sqrt(prec_one), N)
        else:
            z_com = rng.normal(0.0, 1e3, N)

        # Individual posterior about θ_t (each agent uses only their own signal)
        post_prec_ind = X_t + prec_one
        post_mean_ind = prec_one * z_com / max(post_prec_ind, 1e-12)  # prior mean 0
        err_com[t]    = post_mean_ind - theta_t
        pvar_ind[t]   = 1.0 / max(post_prec_ind, 1e-12)
        pvar_col[t]   = 1.0 / max(X_t + prec_agg, 1e-12)

        # ── Idiosyncratic signals ─────────────────────────────────────────────
        theta_i = rng.normal(0.0, 1.0, N)
        prec_pri = lambda_I * e_bar + tau_A

        if prec_pri > 1e-12:
            z_idi = theta_i + rng.normal(0.0, 1.0 / np.sqrt(prec_pri), N)
        else:
            z_idi = rng.normal(0.0, 1e3, N)

        post_prec_idi = sigma_inv2 + prec_pri
        post_mean_idi = prec_pri * z_idi / max(post_prec_idi, 1e-12)
        err_idi[t]    = post_mean_idi - theta_i
        pvar_idi[t]   = 1.0 / max(post_prec_idi, 1e-12)

        X_path[t] = X_t
        e_path[t] = e_bar

        # Advance
        X_t     = F(X_t, tau_A, **kw_)
        theta_t += rng.normal(0.0, np.sqrt(Sigma_sq))

    return dict(X=X_path, effort=e_path,
                err_com=err_com, err_idi=err_idi,
                pvar_ind=pvar_ind, pvar_col=pvar_col, pvar_idi=pvar_idi,
                T=T, N=N)


# ─────────────────────────────────────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Knowledge Collapse ABM", layout="wide")
st.title("Knowledge Collapse — Interactive Explorer")
st.caption("Acemoglu, Kong & Ozdaglar (2026) · Bayesian learning agents on islands")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.page_link("pages/home.py", label="🏠 Back to Home", icon="🏠")
    st.divider()
    st.header("⚙️ Parameters")

    st.subheader("Learning cost")
    alpha = st.slider(
        "Cost steepness  α",
        1.05, 2.0, 1.20, 0.01,
        help="Controls how steeply effort costs rise. "
             "ε = 1/(α−1): elastic (ε > 4) allows collapse; inelastic (ε ≤ 4) does not.")
    eps = 1.0 / (alpha - 1.0)
    if eps > 4:
        st.error(f"ε = {eps:.2f}  →  Elastic regime  ·  collapse possible")
    else:
        st.success(f"ε = {eps:.2f}  →  Inelastic regime  ·  unique stable equilibrium")

    st.subheader("Community")
    N = st.slider(
        "Community size  N",
        5, 500, 50, 5,
        help="Number of agents on the island. "
             "Larger N → stronger collective signal → higher collapse threshold τ_A^c.")
    lambda_G = st.slider(
        "Public learning efficiency  λ_G",
        0.1, 5.0, 1.0, 0.1,
        help="How much each unit of effort contributes to shared knowledge. "
             "Higher λ_G → public good is more valuable.")
    lambda_I = st.slider(
        "Private learning efficiency  λ_I",
        0.1, 5.0, 1.0, 0.1,
        help="How much effort improves an agent's own task knowledge. "
             "Higher λ_I → private return to learning is larger.")

    st.subheader("Knowledge dynamics")
    Sigma_sq = st.slider(
        "Knowledge decay rate  Σ²",
        0.005, 0.2, 0.05, 0.005,
        help="Variance of the random drift in the shared knowledge state each period. "
             "Higher Σ² → knowledge becomes obsolete faster, demanding more refreshing.")
    sigma_inv2 = st.slider(
        "Task prior precision  σ⁻²",
        0.001, 2.0, 0.01, 0.001,
        help="How much agents already know about their task without learning. "
             "Paper baseline: σ² = 1 (σ⁻² = 1). Use 0.01 for visible dynamics.")

    st.subheader("AI capability")
    tau_A = st.slider(
        "AI capability  τ_A",
        0.0, 3.0, 0.0, 0.01,
        help="Precision of the AI signal agents receive about their task θ_{i,t}. "
             "As τ_A rises, agents reduce effort → less shared-knowledge production → erosion.")

    st.divider()
    st.subheader("Simulation")
    T     = st.slider("Time horizon  T", 50, 500, 200, 10)
    X0_lo = st.slider("Low starting point  X₀⁻", 0.001, 2.0,  0.05, 0.01)
    X0_hi = st.slider("High starting point  X₀⁺", 0.1,  19.0, 8.0,  0.1)

# ── shared kwargs ─────────────────────────────────────────────────────────────
kw   = dict(alpha=alpha, lambda_I=lambda_I, lambda_G=lambda_G,
            sigma_inv2=sigma_inv2, N=N, Sigma_sq=Sigma_sq)
kw_e = dict(alpha=alpha, lambda_I=lambda_I, lambda_G=lambda_G,
            sigma_inv2=sigma_inv2)

# ── cached computations ───────────────────────────────────────────────────────
@st.cache_data(max_entries=64)
def cached_tau_c(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq):
    return find_collapse_threshold(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq)

@st.cache_data(max_entries=128)
def cached_ss(tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq):
    return find_steady_states(tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq)

@st.cache_data(max_entries=128)
def cached_welfare_curve(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq):
    tc   = find_collapse_threshold(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq)
    taus = np.linspace(0.0, tc * 1.6, 60)
    w    = [ss_welfare(t, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq)[0] for t in taus]
    return taus, np.array(w), tc

@st.cache_data(max_entries=64)
def cached_scaling(alpha, lambda_I, lambda_G, sigma_inv2, Sigma_sq):
    N_vals = np.array([5, 10, 20, 50, 100, 200, 500])
    tc_vals = [find_collapse_threshold(alpha, lambda_I, lambda_G, sigma_inv2, int(n), Sigma_sq)
               for n in N_vals]
    return N_vals, np.array(tc_vals)

@st.cache_data(max_entries=32)
def cached_sim(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq, seed):
    return _simulate_agents(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq, seed)

tau_c  = cached_tau_c(**kw)
ss     = cached_ss(tau_A, **kw)
pos_ss = [s for s in ss if s > 1e-3]

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Collapse threshold  τ_A^c",  f"{tau_c:.3f}")
c2.metric("Knowledge equilibrium  X̄_h",
          f"{max(pos_ss):.3f}" if pos_ss else "—")
c3.metric("Tipping point  X̄_m",
          f"{min(p for p in pos_ss if p > 1e-3):.4f}" if len(pos_ss) >= 2
          else ("Negligible — safe basin" if tau_A < tau_c * 0.5 else "Near collapse"))
c4.metric("AI status",
          "🔴 COLLAPSED" if tau_A >= tau_c
          else f"✅ {tau_A/tau_c*100:.0f}% of collapse threshold")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📐 Knowledge Dynamics",
    "📈 Time Paths  X_t",
    "💡 Social Welfare  W",
    "📏 Community Size Effect",
    "🧠 Agent Beliefs",
    "📖 Model Guide",
])

# ─── Tab 1: Law of Motion ─────────────────────────────────────────────────────
with tab1:
    st.caption("F(X_t) maps this period's knowledge precision to next period's. "
               "Fixed points are equilibria. Where F lies above the 45° line, knowledge grows; below, it shrinks.")
    col_plot, col_info = st.columns([3, 1])

    with col_plot:
        X_max_plot = min(0.98 / Sigma_sq, max(pos_ss) * 1.5 + 1.0) if pos_ss else 5.0
        Xv = np.concatenate([np.logspace(-6, np.log10(0.1), 100),
                              np.linspace(0.1, X_max_plot, 300)])
        Fv = np.array([F(x, tau_A, **kw) for x in Xv])

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(Xv, Fv, color=C_HIGH, lw=2.2,
                label="Next-period knowledge  F(X_t)")
        ax.plot([0, X_max_plot], [0, X_max_plot], "k--", lw=1.2, alpha=0.45,
                label="No-change line  (X_{t+1} = X_t)")

        for s in ss:
            if s < 1e-3:
                ax.scatter([0], [0], s=100, color=C_ZERO, zorder=5,
                           label="Collapse equilibrium  X=0")
            else:
                dx    = 1e-4
                slope = (F(s + dx, tau_A, **kw) - F(s - dx, tau_A, **kw)) / (2 * dx)
                if slope < 1.0:
                    ax.scatter([s], [s], s=100, color=C_HIGH, zorder=5,
                               label=f"Stable equilibrium  X̄_h = {s:.3f}")
                else:
                    ax.scatter([s], [s], s=100, color=C_MID, zorder=5,
                               marker="^", label=f"Tipping point  X̄_m = {s:.4f}")

        if tau_A > 0:
            ax.axvline(tau_c, color=C_AI, lw=1.2, ls=":", alpha=0.6,
                       label=f"Collapse threshold  τ_A^c = {tau_c:.3f}")

        ax.set_xlim(0, X_max_plot)
        ax.set_ylim(0, X_max_plot)
        ax.set_xlabel("Current knowledge precision  X_t", fontsize=11)
        ax.set_ylabel("Next-period knowledge precision  X_{t+1}", fontsize=11)
        ax.set_title(f"Knowledge Law of Motion  ·  AI capability τ_A = {tau_A:.3f}", fontsize=12)
        handles, labels_l = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels_l):
            if l not in seen: seen[l] = h
        ax.legend(seen.values(), seen.keys(), fontsize=8)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)

    with col_info:
        st.markdown("**Equilibria**")
        for s in ss:
            if s < 1e-3:
                st.markdown(f"- X = 0  ·  {'🔴 stable (collapse)' if eps > 4 else '⚪ unstable'}")
            else:
                dx    = 1e-4
                slope = (F(s + dx, tau_A, **kw) - F(s - dx, tau_A, **kw)) / (2 * dx)
                icon  = "🔵 stable" if slope < 1.0 else "🟠 tipping point"
                st.markdown(f"- X = {s:.3f}  ·  {icon}")

        st.divider()
        st.markdown("**At equilibrium X̄_h**")
        if pos_ss:
            Xh    = max(pos_ss)
            e_bar = solve_effort(Xh, tau_A, **kw_e)
            Y_bar = sigma_inv2 + lambda_I * e_bar + tau_A
            st.metric("Effort  ē*",                  f"{e_bar:.4f}")
            st.metric("Task precision  Ȳ",            f"{Y_bar:.3f}")
            st.metric("Shared knowledge quality  G(X̄_h)", f"{float(G(Xh)):.3f}")
            st.metric("Task knowledge quality  G(Ȳ)",      f"{float(G(Y_bar)):.3f}")

# ─── Tab 2: Time Series ───────────────────────────────────────────────────────
with tab2:
    st.caption("Each line is a deterministic trajectory from a different starting point. "
               "Low starts collapse; high starts reach the knowledge equilibrium. "
               "Horizontal lines mark equilibria.")
    n_traj   = st.slider("Additional random trajectories", 0, 8, 3)
    rng_seed = st.number_input("Random seed", 0, 9999, 42)

    @st.cache_data(max_entries=64)
    def run_trajectories(X0s, T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq):
        kw_ = dict(alpha=alpha, lambda_I=lambda_I, lambda_G=lambda_G,
                   sigma_inv2=sigma_inv2, N=N, Sigma_sq=Sigma_sq)
        paths = []
        for X0 in X0s:
            path = [float(X0)]
            for _ in range(T - 1):
                path.append(F(path[-1], tau_A, **kw_))
            paths.append(np.array(path))
        return paths

    rng_ui    = np.random.default_rng(int(rng_seed))
    X0_extras = rng_ui.uniform(X0_lo, X0_hi, n_traj).tolist()
    X0s_all   = [X0_lo, X0_hi] + X0_extras
    paths     = run_trajectories(tuple(X0s_all), T, tau_A, **kw)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (X0, path) in enumerate(zip(X0s_all, paths)):
        col = C_ZERO if i == 0 else (C_HIGH if i == 1 else "gray")
        lw  = 2.0 if i < 2 else 1.0
        lbl = (f"X₀={X0:.2f}  (below tipping point — collapses)"
               if i == 0 else
               f"X₀={X0:.2f}  (above tipping point — stable)"
               if i == 1 else f"X₀={X0:.2f}")
        ax.plot(path, color=col, lw=lw, alpha=0.8 if i >= 2 else 1.0, label=lbl)

    for s in ss:
        if s > 0.05:
            slope = (F(s + 1e-4, tau_A, **kw) - F(s - 1e-4, tau_A, **kw)) / 2e-4
            ax.axhline(s, color=C_HIGH if slope < 1.0 else C_MID,
                       lw=1.2, ls="-" if slope < 1.0 else "--", alpha=0.45)

    ax.set_xlabel("Period  t", fontsize=11)
    ax.set_ylabel("Shared knowledge precision  X_t", fontsize=11)
    ax.set_title(f"Knowledge trajectories  ·  AI capability τ_A = {tau_A:.3f}", fontsize=12)
    ax.legend(fontsize=8, ncol=1)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Final knowledge level after T periods**")
    cols_t = st.columns(min(len(X0s_all), 5))
    for i, (X0, path) in enumerate(zip(X0s_all[:5], paths[:5])):
        cols_t[i].metric(f"X₀ = {X0:.2f}", f"{path[-1]:.3f}",
                         delta=f"{path[-1] - path[0]:+.3f}")

# ─── Tab 3: Welfare ───────────────────────────────────────────────────────────
with tab3:
    st.caption("Social welfare W = G(X̄_h)·G(Ȳ) − ē^α/α. "
               "As AI capability τ_A rises, task-knowledge quality G(Ȳ) improves "
               "but shared-knowledge quality G(X̄_h) erodes. The net effect is hump-shaped.")
    taus_w, w_arr, tc_w = cached_welfare_curve(**kw)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(taus_w, w_arr, color=C_WELF, lw=2.2)
    ax.axvline(tc_w, color=C_ZERO, lw=1.5, ls="--",
               label=f"Collapse threshold  τ_A^c = {tc_w:.3f}")
    ax.axvline(tau_A, color=C_AI, lw=1.8, alpha=0.8,
               label=f"Current AI capability  τ_A = {tau_A:.3f}")
    if w_arr.max() > 0:
        peak_idx = int(np.argmax(w_arr))
        tau_star = taus_w[peak_idx]
        ax.axvline(tau_star, color=C_MID, lw=1.5, ls=":",
                   label=f"Welfare-maximising level  τ_A* = {tau_star:.3f}")
        ax.scatter([tau_star], [w_arr[peak_idx]], color=C_MID, s=80, zorder=5)
    ax.set_xlabel("AI capability  τ_A", fontsize=11)
    ax.set_ylabel("Social welfare  W", fontsize=11)
    ax.set_title("Welfare vs AI capability", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    taus_ss = taus_w[taus_w <= tc_w * 1.01]
    Xh_vals = [ss_welfare(t, **kw)[1] for t in taus_ss]
    GXh     = G(np.array(Xh_vals))
    ax2.plot(taus_ss, Xh_vals, color=C_HIGH, lw=2.0,
             label="Knowledge equilibrium  X̄_h")
    ax2r = ax2.twinx()
    ax2r.plot(taus_ss, GXh, color=C_MID, lw=1.8, ls="--",
              label="Shared knowledge quality  G(X̄_h)")
    ax2.axvline(tau_A, color=C_AI,  lw=1.5, alpha=0.7)
    ax2.axvline(tc_w,  color=C_ZERO, lw=1.3, ls="--", alpha=0.5)
    ax2.set_xlabel("AI capability  τ_A", fontsize=11)
    ax2.set_ylabel("Knowledge equilibrium  X̄_h", fontsize=11, color=C_HIGH)
    ax2r.set_ylabel("Shared knowledge quality  G(X̄_h)", fontsize=11, color=C_MID)
    ax2.set_title("Knowledge erosion path", fontsize=12)
    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, fontsize=9)
    ax2.grid(True, alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Welfare decomposition at current τ_A**")
    if pos_ss:
        Xh    = max(pos_ss)
        e_bar = solve_effort(Xh, tau_A, **kw_e)
        Y_bar = sigma_inv2 + lambda_I * e_bar + tau_A
        w_now = float(G(Xh) * G(Y_bar) - e_bar ** alpha / alpha)
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Shared knowledge quality  G(X̄_h)", f"{float(G(Xh)):.4f}")
        dc2.metric("Task knowledge quality  G(Ȳ)",      f"{float(G(Y_bar)):.4f}")
        dc3.metric("Effort cost  ē^α/α",                f"{e_bar**alpha/alpha:.4f}")
        dc4.metric("Net welfare  W",                    f"{w_now:.4f}")
    else:
        st.warning("No high equilibrium at current τ_A — system has collapsed.")

# ─── Tab 4: Scaling Law ───────────────────────────────────────────────────────
with tab4:
    st.caption("Proposition 6: the collapse threshold τ_A^c grows logarithmically in community size N. "
               "Larger communities produce stronger collective signals, making knowledge more resilient.")
    N_plot, tc_plot = cached_scaling(alpha, lambda_I, lambda_G, sigma_inv2, Sigma_sq)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.scatter(N_plot, tc_plot, color=C_HIGH, s=70, zorder=5)
    ax.plot(N_plot,    tc_plot, color=C_HIGH, lw=1.5, alpha=0.5)
    ax.axhline(tau_A,  color=C_AI, lw=1.5, ls="--",
               label=f"Current AI capability  τ_A = {tau_A:.3f}")
    safe = N_plot[tc_plot >  tau_A]
    dang = N_plot[tc_plot <= tau_A]
    if len(safe): ax.axvspan(safe[0], safe[-1], alpha=0.07, color=C_HIGH)
    if len(dang): ax.axvspan(max(dang[0], 1), dang[-1], alpha=0.10, color=C_ZERO)
    ax.set_xlabel("Community size  N", fontsize=11)
    ax.set_ylabel("Collapse threshold  τ_A^c", fontsize=11)
    ax.set_title("How community size protects knowledge", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.scatter(np.log(N_plot), tc_plot, color=C_HIGH, s=70, zorder=5)
    c_fit = np.polyfit(np.log(N_plot), tc_plot, 1)
    xf    = np.linspace(np.log(N_plot[0]), np.log(N_plot[-1]), 100)
    ax2.plot(xf, np.polyval(c_fit, xf), color=C_MID, lw=2.0, ls="--",
             label=f"{c_fit[0]:.3f}·ln(N) + {c_fit[1]:.3f}")
    ax2.axhline(tau_A, color=C_AI, lw=1.5, ls="--",
                label=f"Current AI capability  τ_A = {tau_A:.3f}")
    r2 = 1 - np.var(tc_plot - np.polyval(c_fit, np.log(N_plot))) / (np.var(tc_plot) + 1e-12)
    ax2.set_xlabel("Log community size  ln(N)", fontsize=11)
    ax2.set_ylabel("Collapse threshold  τ_A^c", fontsize=11)
    ax2.set_title("Log-linear fit", fontsize=12)
    ax2.legend(title=f"R² = {r2:.4f}", fontsize=9)
    ax2.grid(True, alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Collapse threshold by community size**")
    cols_s = st.columns(len(N_plot))
    for i, (n, tc_) in enumerate(zip(N_plot, tc_plot)):
        cols_s[i].metric(f"N = {int(n)}", f"{tc_:.3f}",
                         delta="✅ safe" if tc_ > tau_A else "🔴 collapsed")

# ─── Tab 5: Agent Beliefs ─────────────────────────────────────────────────────
with tab5:
    st.caption(
        "Each dot is one agent's Bayesian **belief error** after observing signals. "
        "**Horizontal axis**: how far each agent's posterior mean about the *shared knowledge state θ_t* "
        "is from the truth — using only their *own* signal (precision λ_G·ē). "
        "**Vertical axis**: how far each agent's posterior mean about their *task state θ_{i,t}* "
        "is from the truth. "
        "A tight cloud near the origin means agents know both dimensions well. "
        "The **purple ellipse** is the individual 1-σ posterior; "
        "the **teal ellipse** is the collective 1-σ posterior (public good) — "
        "their gap illustrates the knowledge externality."
    )

    c_ctrl, c_plot = st.columns([1, 3])
    with c_ctrl:
        sim_T    = st.slider("Simulation length  T", 20, 300, 80, 10)
        sim_seed = st.number_input("Seed", 0, 9999, 42, key="sim_seed")

    sim    = cached_sim(sim_T, tau_A, alpha, lambda_I, lambda_G,
                        sigma_inv2, N, Sigma_sq, int(sim_seed))
    t_max  = sim_T - 1

    with c_ctrl:
        period = st.slider("Period  t", 0, t_max, 0)

    ec   = sim["err_com"][period]
    ei   = sim["err_idi"][period]
    e_t  = sim["effort"][period]
    X_t  = sim["X"][period]
    s_ind = np.sqrt(sim["pvar_ind"][period])
    s_col = np.sqrt(sim["pvar_col"][period])
    s_idi = np.sqrt(sim["pvar_idi"][period])

    with c_plot:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                                 gridspec_kw={"width_ratios": [3, 2]})

        # ── left: belief cloud ────────────────────────────────────────────────
        ax = axes[0]
        e_max_plot = max(sim["effort"].max(), 1e-8)
        sc = ax.scatter(ec, ei, c=np.full(N, e_t), cmap="YlOrRd",
                        s=45, alpha=0.75, vmin=0, vmax=e_max_plot, zorder=3)
        ax.scatter([0], [0], marker="*", s=250, color="red", zorder=10,
                   label="True state  (θ_t, θ_{i,t})")
        ax.axhline(0, color="gray", lw=0.6, alpha=0.4)
        ax.axvline(0, color="gray", lw=0.6, alpha=0.4)

        # Individual posterior ellipse
        ell_ind = Ellipse((0, 0), 2 * s_ind, 2 * s_idi,
                          fill=False, edgecolor=C_IND, lw=2.0, ls="--",
                          label=f"Individual 1σ  ({s_ind:.3f}, {s_idi:.3f})")
        ax.add_patch(ell_ind)

        # Collective posterior ellipse (x only tighter)
        ell_col = Ellipse((0, 0), 2 * s_col, 2 * s_idi,
                          fill=False, edgecolor=C_COL, lw=2.0, ls="-",
                          label=f"Collective 1σ  ({s_col:.3f}, {s_idi:.3f})")
        ax.add_patch(ell_col)

        lim = max(3 * s_ind, np.abs(ec).max() * 1.2, np.abs(ei).max() * 1.2, 0.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        plt.colorbar(sc, ax=ax, label="Effort ē  (warmer = more learning)")
        ax.set_xlabel("Shared-knowledge belief error  (posterior mean − θ_t)", fontsize=10)
        ax.set_ylabel("Task-knowledge belief error  (posterior mean − θ_{i,t})", fontsize=10)
        ax.set_title(
            f"Period {period}  ·  X_t = {X_t:.3f}  ·  ē = {e_t:.5f}",
            fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

        # ── right: knowledge path ─────────────────────────────────────────────
        ax2 = axes[1]
        ax2.plot(sim["X"], color=C_HIGH, lw=1.5,
                 label="Shared knowledge  X_t")
        ax2.axvline(period, color=C_AI, lw=2.0, alpha=0.8)
        ax2.scatter([period], [X_t], color=C_AI, s=80, zorder=5,
                    label=f"Current period  t={period}")
        # shade collective vs individual precision gap
        ax2r = ax2.twinx()
        ind_std = np.sqrt(sim["pvar_ind"])
        col_std = np.sqrt(sim["pvar_col"])
        ax2r.fill_between(range(sim_T), col_std, ind_std,
                          alpha=0.18, color=C_IND,
                          label="Externality gap  (individual − collective σ)")
        ax2r.set_ylabel("Belief std-dev  (1/√precision)", fontsize=9, color=C_IND)
        ax2r.tick_params(axis="y", labelcolor=C_IND)
        ax2.set_xlabel("Period  t", fontsize=10)
        ax2.set_ylabel("Shared knowledge precision  X_t", fontsize=10)
        ax2.set_title("Knowledge path  +  externality gap", fontsize=11)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

    # Summary row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Shared knowledge  X_t",            f"{X_t:.3f}")
    m2.metric("Effort  ē",                        f"{e_t:.5f}")
    m3.metric("Individual uncertainty  σ_ind",    f"{s_ind:.3f}")
    m4.metric("Collective uncertainty  σ_col",    f"{s_col:.3f}")
    m5.metric("Externality gap  σ_ind − σ_col",   f"{s_ind - s_col:.3f}")

# ─── Tab 6: Model Guide ───────────────────────────────────────────────────────
with tab6:
    st.markdown("## Model Guide")
    st.markdown(
        "Complete reference for the baseline replication model. "
        "Each section covers the **economic intuition**, the **formal definition**, "
        "and **what happens when you change the parameter**. "
        "Read sequentially — later sections assume earlier ones."
    )
    st.divider()

    with st.expander("🌍  States of the world  —  θ_t  and  θ_{i,t}", expanded=True):
        st.markdown("""
The model has **two distinct knowledge objects** agents try to learn about.
Understanding why they are separate is the key to understanding the whole model.

---
**θ_t — Shared knowledge state  (common, persistent, drifting)**

The true value of the knowledge problem the entire community is working on.
Think of it as the "true state of the field" — the answer to the shared scientific question.
It follows a **random walk**: θ_{t+1} = θ_t + ε_t,  ε_t ~ N(0, Σ²).
The drift (Σ²) means the question slowly evolves — last decade's understanding is partially obsolete.
Agents **never observe θ_t directly**: they infer it from signals generated by each other's effort.

The community's current precision about θ_t is X_t — the key state variable of the model.

---
**θ_{i,t} — Task-specific state  (private, i.i.d., resets every period)**

The true value of agent *i*'s idiosyncratic problem in period *t*.
θ_{i,t} ~ N(0, 1), drawn **fresh every period**, uncorrelated across agents and time.

**Why does task knowledge reset?**
Tasks are fundamentally idiosyncratic. A lawyer's client changes every case;
a doctor's patient is different every appointment; an engineer faces a different
specification on each project. The context cannot be shared and is useless once the task is done.
This is *different* from domain knowledge: knowing the law in general (θ_t, persistent, shared)
is not the same as knowing this particular client's situation (θ_{i,t}, private, ephemeral).

In AI terms: domain knowledge is the model's weights (slow to build, widely shared).
Task knowledge is the context window (instant to fill, agent-specific, discarded after inference).

**AI (τ_A) substitutes only for task knowledge** — it fills the context window — not for domain knowledge.

| Symbol | What it represents | Persistent? | Shared? |
|---|---|---|---|
| **θ_t** | State of the shared knowledge problem | Yes (random walk) | Yes — same for all agents |
| **θ_{i,t}** | Agent i's current task context | No — reset each period | No — private to agent i |
| **Σ²** | Rate at which the shared problem evolves | — | Controls X_t decay rate |
""")

    with st.expander("📡  Bayesian precision, signals, and what AI does  —  X_t, Y_{i,t}, τ_A"):
        st.markdown("""
Agents hold **Gaussian beliefs**. A belief with precision τ has variance 1/τ.
Higher precision = tighter posterior = better knowledge.
Precisions from **independent sources add**: prior precision π₀ + signal precision π₁ = posterior precision π₀ + π₁.

---
**X_t — Shared knowledge precision  (the key state variable)**

The community's collective posterior precision about θ_t at the start of period t.
The entire model dynamics live in X_t. Maximum value = 1/Σ².

**The strategic role of X_t — the Leontief complement:**
Output = G(X_t) · G(Y_{i,t}).  If X_t = 0, output = 0 regardless of how good task knowledge is.
This means agents have high private returns to effort *only when X_t is high* —
making X_t a strategic complement: high X → high effort → high X (self-reinforcing).
Low X → low effort → low X (also self-reinforcing — the collapse spiral).

---
**Y_{i,t} — Task knowledge precision  (per-agent, built each period)**

Y_{i,t} = σ⁻² + λ_I·e_{i,t} + τ_A

Three independent sources (precisions add):
- σ⁻²: prior precision (baseline knowledge without any learning)
- λ_I·e_{i,t}: private learning signal (precision proportional to effort)
- τ_A: AI signal (a **free** private signal requiring zero effort)

Because task knowledge resets, agents must rebuild Y_{i,t} each period.

---
**τ_A — AI capability  (the central parameter)**

τ_A is the precision of the AI signal. It acts exactly like a free private signal about θ_{i,t}.
**AI does not produce shared knowledge**: it raises Y_{i,t} but has no direct effect on X_t.

**Why AI causes collapse:**
1. Higher τ_A → Y_{i,t} is already high without effort → agents reduce effort ē*
2. Less effort → weaker public signal λ_G·N·ē → X_{t+1} falls
3. Lower X_{t+1} → G(X) falls → private return to effort falls → agents reduce effort further
4. Past τ_A^c: this feedback spiral cannot stabilise → X_t → 0

AI is not uniquely bad — it is that AI *substitutes for effort in the one activity*
(private learning) that also produces shared knowledge as a public by-product.

| Symbol | Source | Enters | Resets? |
|---|---|---|---|
| **σ⁻²** | Prior (structural) | Y_{i,t} | Same every period |
| **λ_I·e** | Private effort | Y_{i,t} | Yes (effort chosen each period) |
| **τ_A** | AI signal (free) | Y_{i,t} | Yes (received each period) |
| **λ_G·N·ē** | Collective effort | X_{t+1} | No — persistent public good |
""")

    with st.expander("💪  Effort, the FOC, and the credit mechanism  —  e*, α, λ_G, λ_I"):
        st.markdown("""
**Each agent chooses effort e to maximise expected output minus cost:**

max_e  G(X_t) · G(Y_{i,t}(e)) − e^α / α

where Y_{i,t}(e) = σ⁻² + λ_I·e + τ_A.

**First-order condition (FOC):**

λ_I · G(X_t) · g(σ⁻² + λ_I·e + τ_A)  =  e^{α−1}

Left side: marginal benefit — how much better the private signal becomes (λ_I·g(Y))
times how much that matters given the shared knowledge quality G(X_t).
Right side: marginal cost — increasing because α > 1.

**The credit mechanism:**
G(X_t) acts as a multiplier on the private return to effort.
When shared knowledge is high (G(X_t) near 1), good task knowledge is highly productive.
When shared knowledge collapses (G(X_t) → 0), even perfect task knowledge is useless —
the Leontief product goes to zero. This is why agents reduce effort when X_t falls:
the *credit* to effort (its productive value) depends on the shared knowledge context.

**The public-good externality in the FOC:**
Agents choose e based only on λ_I (private learning efficiency), NOT on λ_G·N (aggregate public good).
The public signal λ_G·e per agent aggregates to λ_G·N·ē, but each agent treats their
contribution as negligible (1/N → 0). This is the fundamental under-investment:
optimal effort would account for λ_G·N, but equilibrium effort accounts only for λ_I.

---
| Symbol | Role | Effect of increase |
|---|---|---|
| **e_{i,t}** | Effort chosen by agent i | More effort → better Y (private) AND stronger public signal (social) |
| **ē*** | Equilibrium effort (FOC solution) | — |
| **α** | Cost steepness (C(e) = e^α/α) | Higher → more expensive effort → lower ē* → lower X̄_h |
| **ε = 1/(α−1)** | Effort elasticity (regime switch at ε = 4) | — |
| **λ_G** | Public learning efficiency | Higher → stronger public good → higher X̄_h → later τ_A^c |
| **λ_I** | Private learning efficiency | Higher → greater private return → more effort → higher X̄_h as side effect |
""")

    with st.expander("⚙️  Production, the Leontief structure, and welfare  —  G(τ), F(X), W"):
        st.markdown("""
**G(τ) = 2Φ(√τ) − 1 — the knowledge quality function**

G(τ) is the probability that an agent with posterior precision τ correctly identifies
the sign of the true state (a binary ±1 task).
G(0) = 0 (no knowledge → random guess), G(∞) = 1 (perfect knowledge → always correct).
G is **concave** (diminishing returns to precision) and **bounded** in [0,1].

**Output = G(X_t) · G(Y_{i,t}) — the Leontief complement**

This is the structural core of the model.
Output requires BOTH shared domain knowledge G(X_t) AND task-specific knowledge G(Y_{i,t}).
If either component is zero, output is zero — they are complements, not substitutes.

Compare:
- Substitutes: output = G(X) + G(Y) — AI improves Y, X collapses, net effect ambiguous
- **Leontief complements: output = G(X) · G(Y) — AI improves Y, but if X → 0, gains vanish**

This is what makes AI dangerous: even if AI makes task knowledge perfect (G(Y) = 1),
collapse of shared knowledge (G(X) → 0) drives output to zero.
The short-run gain from AI is overwhelmed by the long-run loss of domain knowledge.

**F(X_t) — the law of motion**

X_{t+1} = F(X_t) = [(X_t + λ_G·N·ē*(X_t))⁻¹ + Σ²]⁻¹

Step 1: X_t (prior precision) + λ_G·N·ē* (new aggregate public signal) = posterior precision
Step 2: Posterior erodes by drift Σ² → next period's prior

F maps this period's knowledge to next period's. Fixed points F(X*) = X* are equilibria.
The Tab 1 diagram plots F vs X: above the 45° line → knowledge grows; below → shrinks.

**W — social welfare at the high equilibrium**

W = G(X̄_h) · G(Ȳ) − ē*^α / α

As τ_A rises: G(Ȳ) improves (AI helps tasks ✓), G(X̄_h) falls (knowledge erodes ✗),
cost ē*^α/α falls (less effort needed). Net effect: **hump-shaped** — welfare first rises then falls,
reaching zero at τ_A^c. The welfare-maximising τ_A* is interior to (0, τ_A^c).
""")

    with st.expander("⚖️  Equilibria and the collapse threshold  —  X̄_h, X̄_m, τ_A^c"):
        st.markdown("""
Fixed points of F are equilibria. Stability: F′(X*) < 1 → stable; F′(X*) > 1 → unstable.

**X̄_h — High-knowledge equilibrium  (stable)**
Largest positive fixed point of F. If X_0 > X̄_m, system converges here.
**Decreases** monotonically as τ_A rises — even before collapse, AI gradually erodes X̄_h.

**X̄_m — Tipping point  (unstable, elastic regime only)**
Middle fixed point where F′ > 1. Basin boundary: above → converge to X̄_h; below → collapse to 0.
Very small when τ_A is small (safe basin is almost the whole positive real line).
Grows to meet X̄_h at τ_A^c (saddle-node bifurcation).

**X = 0 — Collapse equilibrium**
In elastic regime (ε > 4): locally stable — once below X̄_m, stays at zero forever.
In inelastic regime (ε ≤ 4): unstable — knowledge recovers from any positive start.

**τ_A^c — Collapse threshold  (the key policy boundary)**
Critical level where X̄_m merges with X̄_h. Above τ_A^c: no positive equilibrium, collapse is certain.
τ_A^c grows **logarithmically in N**: to handle twice the AI capability, you need exponentially
more agents. The "grow your community" solution has diminishing returns against improving AI.

**Why log(N) scaling matters for policy:**
If AI capability grows linearly over time (plausible), required community size grows exponentially.
No amount of "adding more researchers" can permanently protect knowledge — only delay collapse.
The correct long-run solution: limit AI capability (garbling) or cross-domain renewal (extension model).
""")

    with st.expander("🔀  Regime classification  —  why ε = 4 is the critical value"):
        st.markdown("""
The **effort elasticity ε = 1/(α−1)** determines the qualitative structure of the model.

| Elasticity | α range | X = 0 stability | Steady states | Collapse possible? |
|---|---|---|---|---|
| **ε ≤ 4**  (inelastic) | α ≥ 1.25 | Unstable | Unique positive stable X̄_h | No |
| **ε > 4**  (elastic) | α < 1.25 | Stable | Three: 0 (stable), X̄_m (unstable), X̄_h (stable) | Yes |

**Why ε = 4?**

Near X = 0, equilibrium effort scales as ē*(X) ~ X^{ε/2}.
The public signal is λ_G·N·ē*(X) ~ X^{ε/2}.
For F(X) < X near zero (X = 0 is stable), we need the total update X + λ_G·N·ē*(X) to grow
slower than 1/Σ² after inversion. Working through the algebra, the condition is ε/2 > 2, i.e., **ε > 4**.

Below ε = 4: even at very low X, agents maintain enough effort to sustain collective learning.
Above ε = 4: effort collapses so fast as X → 0 that the collective signal vanishes.
The self-reinforcing spiral — low X → low effort → even lower X — reaches zero.
""")

    with st.expander("🤝  The public-good externality — why markets under-invest"):
        st.markdown("""
**The central market failure in the model.**

Effort e produces two simultaneous outputs:
1. **Private**: signal about θ_{i,t} with precision λ_I·e — agent keeps this.
2. **Public**: signal about θ_t with precision λ_G·e — aggregates to λ_G·N·ē for everyone.

The FOC only captures the *private* return (λ_I·g(Y)·G(X)).
The *social* return also includes λ_G·N·g(X)·G(Y) — N times larger.
Since each agent's contribution is 1/N → 0 of the aggregate, agents free-ride.

**Consequences:**
- Equilibrium X̄_h is lower than socially optimal
- Collapse threshold τ_A^c is lower than it could be under a social planner
- The welfare loss from AI is amplified by the externality

**Agent Beliefs tab makes this visible:**
- Purple ellipse (individual 1σ): each agent's uncertainty using their own signal (precision X + λ_G·ē)
- Teal ellipse (collective 1σ): community's aggregate uncertainty (precision X + λ_G·**N**·ē)
- Gap between ellipses = externality gap — knowledge that exists collectively but no one fully internalises

The gap is what a Pigouvian subsidy on learning effort would close.
""")

    with st.expander("🏛️  Optimal policy  —  garbling the AI signal"):
        st.markdown("""
**What should a regulator do?**

The paper (Proposition 8) characterises the optimal policy as **garbling**:
reduce the AI signal precision agents observe from τ_A to τ̃_A ≤ τ_A.

**Two-phase optimal policy:**

1. τ_A ≤ τ_A*: Allow full AI — welfare gains from better task knowledge outweigh erosion of X.
2. τ_A > τ_A*: Cap at τ_A* — suppress AI precision to preserve shared knowledge.

τ_A* is the welfare-maximising AI level (peak of the W(τ_A) curve, red line in Social Welfare tab).
In most parameterisations τ_A* < τ_A^c: the welfare peak is strictly inside the stable region.

**Why garbling, not banning?**
At low τ_A, AI is unambiguously welfare-improving. A blanket ban forgoes these gains.
The optimal policy *allows* AI up to τ_A*, then holds it there.

**The key insight:**
Without regulation, the market equilibrium overshoots τ_A* because agents don't internalise
the damage to X from reduced effort. The externality biases private AI adoption decisions
toward too much AI. Optimal regulation does not ban AI — it caps precision at τ_A*.
""")

    with st.expander("🗂️  Quick-reference  —  all symbols at a glance"):
        st.markdown("""
| Symbol | Plain English | Type | Resets each period? |
|---|---|---|---|
| **θ_t** | True shared knowledge state (random walk) | Exogenous state | No (drifts) |
| **θ_{i,t}** | True task state for agent i (i.i.d.) | Exogenous state | Yes |
| **X_t** | Shared knowledge precision = 1/Var[belief about θ_t] | Endogenous state | No (law of motion) |
| **Y_{i,t}** | Task-knowledge precision = σ⁻² + λ_I·e + τ_A | Derived | Yes |
| **e_{i,t} / ē*** | Individual effort / equilibrium effort | Choice variable | Yes |
| **α** | Effort cost steepness  C(e) = e^α/α | Parameter | — |
| **ε = 1/(α−1)** | Effort elasticity  (regime switch at ε = 4) | Derived parameter | — |
| **N** | Community size | Parameter | — |
| **λ_G** | Public learning efficiency  (effort → shared signal) | Parameter | — |
| **λ_I** | Private learning efficiency  (effort → private signal) | Parameter | — |
| **Σ²** | Knowledge decay rate  (drift variance per period) | Parameter | — |
| **σ⁻²** | Task prior precision  (baseline knowledge without learning) | Parameter | — |
| **τ_A** | AI capability  (precision of free AI signal about θ_{i,t}) | Policy variable | — |
| **G(τ) = 2Φ(√τ)−1** | Knowledge quality  (probability of correct sign, [0,1]) | Model function | — |
| **g(τ) = G′(τ)** | Marginal knowledge quality  (appears in FOC) | Model function | — |
| **F(X_t)** | Law of motion:  X_{t+1} = [(X_t + λ_G·N·ē*)⁻¹ + Σ²]⁻¹ | Model function | — |
| **X̄_h** | High-knowledge equilibrium  (stable) | Endogenous | — |
| **X̄_m** | Tipping point  (unstable, elastic regime only) | Endogenous | — |
| **τ_A^c** | Collapse threshold  (saddle-node bifurcation point) | Endogenous | — |
| **τ_A*** | Welfare-maximising AI capability  (optimal policy target) | Policy optimum | — |
| **W** | Social welfare:  G(X̄_h)·G(Ȳ) − ē*^α/α | Welfare measure | — |
""")


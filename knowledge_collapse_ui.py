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
        "Complete reference for every variable, function, and proposition in the model. "
        "Each entry gives the **plain-English meaning**, the **formal definition**, "
        "the **units / range**, and **what happens when you increase it**."
    )
    st.divider()

    # ── States of the world ───────────────────────────────────────────────────
    with st.expander("🌍  States of the world  —  θ_t, θ_{i,t}", expanded=True):
        st.markdown("""
| Symbol | Name | Definition | Range | Effect of increase |
|---|---|---|---|---|
| **θ_t** | Shared knowledge state | The true value of general / domain knowledge in period *t*. Follows a random walk: θ_{t+1} = θ_t + ε_t, ε_t ~ N(0, Σ²). Agents never observe it directly; they infer it from signals. | ℝ (drifts over time) | Larger |θ_t| means the true state has drifted further; more learning is needed to track it. |
| **θ_{i,t}** | Task-specific state | The true value of agent *i*'s idiosyncratic context in period *t*. Drawn fresh every period: θ_{i,t} ~ N(0, 1). Uncorrelated across agents and over time — "context" that cannot be shared. | ℝ, drawn ~ N(0,1) | N/A — exogenous random draw each period. |
| **Σ²** | Knowledge decay rate | Variance of the random drift in θ_t each period. Higher Σ² means shared knowledge becomes obsolete faster and more effort is needed to refresh it. | (0, ∞), slider: 0.005–0.20 | Faster drift → lower steady-state X̄_h → lower welfare → collapse threshold τ_A^c falls. |
""")

    # ── Precision and beliefs ─────────────────────────────────────────────────
    with st.expander("📡  Precision, signals, and Bayesian beliefs  —  X_t, Y_{i,t}, σ⁻², τ_A"):
        st.markdown("""
Agents have **Gaussian beliefs**. A belief with precision τ has variance 1/τ.
Higher precision = tighter posterior = better knowledge.

| Symbol | Name | Definition | Range | Effect of increase |
|---|---|---|---|---|
| **X_t** | Shared knowledge precision | The precision of the community's collective posterior about θ_t at the start of period *t*. It is the key state variable of the model: the entire dynamics live in X_t. Maximum possible value = 1/Σ². | [0, 1/Σ²] | Higher X_t → agents value private learning more (Leontief complement) → more effort → self-reinforcing high-knowledge equilibrium. |
| **Y_{i,t}** | Task-knowledge precision | Posterior precision of agent *i* about their own θ_{i,t}. Composed of: prior σ⁻² + private learning λ_I·e + AI signal τ_A. | [σ⁻², ∞) | Higher Y → better task performance → higher output G(Y). |
| **σ⁻²** | Task prior precision | How much agents already know about their task without any learning or AI. If σ⁻² is large, even without effort agents have a decent estimate of θ_{i,t}; the marginal value of learning is small. Paper baseline: σ² = 1 (σ⁻² = 1). We use σ⁻² = 0.01 for visible dynamics. | (0, ∞), slider: 0.001–2.0 | Higher σ⁻² → equilibrium effort is lower → weaker public-good production → lower X̄_h. With σ⁻² = 0 the regime bifurcation breaks down numerically. |
| **τ_A** | AI capability | Precision of the AI signal each agent receives about their task θ_{i,t}. Acts exactly like a free private signal — it directly substitutes for human private learning. Critically, **AI does not produce shared knowledge**: it adds to Y but not to X. | [0, ∞), slider: 0–3.0 | As τ_A rises, agents need less effort for task knowledge → effort falls → public signal (λ_G·N·e) shrinks → X_{t+1} falls. Past τ_A^c this feedback collapses shared knowledge entirely. |
""")

    # ── Effort and learning ───────────────────────────────────────────────────
    with st.expander("💪  Effort, learning efficiencies, and cost  —  e, α, λ_G, λ_I"):
        st.markdown("""
| Symbol | Name | Definition | Range | Effect of increase |
|---|---|---|---|---|
| **e_{i,t}** | Learning effort | The amount of cognitive effort agent *i* exerts in period *t*. A scalar chosen to maximise expected output minus cost. Each unit of effort simultaneously improves the private signal (precision λ_I·e) **and** contributes to the public signal (precision λ_G·e per agent). | [0, ∞) | More effort → better private and public learning, but higher cost. |
| **ē*** | Equilibrium effort | The unique solution to the first-order condition (FOC) at given X_t and τ_A. All agents choose the same ē* because they are ex-ante identical. Since each agent's contribution to the public good is infinitesimal, they **ignore the public-good value** of their effort — a classic externality. | Computed | — |
| **α** | Cost steepness | Curvature of the effort cost function C(e) = e^α/α. With α > 1, marginal cost is increasing. The **elasticity of effort** is ε = 1/(α−1); this elasticity determines which regime the model is in. | (1, ∞), slider: 1.05–2.0 | Higher α → effort is more expensive → less effort in equilibrium → lower X̄_h. Crucially: α < 1.25 (ε > 4) → elastic → collapse possible. |
| **ε = 1/(α−1)** | Effort elasticity | How sensitive equilibrium effort is to the value of private learning. **ε > 4 (α < 1.25)**: elastic regime — the collapse steady state at X=0 is locally stable, and multiple equilibria can exist. **ε ≤ 4 (α ≥ 1.25)**: inelastic regime — unique stable high-knowledge equilibrium, no collapse. | (0, ∞) | — |
| **λ_G** | Public learning efficiency | How much one unit of effort by one agent contributes to the precision of the public signal about θ_t. The **aggregate** public signal precision is λ_G·N·ē. This is the channel through which effort becomes a public good: private effort has a social return λ_G·N times bigger than the private return, but agents only internalise the private part. | (0, ∞), slider: 0.1–5.0 | Higher λ_G → stronger public-good externality → more underinvestment in a laissez-faire equilibrium → but also higher X̄_h because each unit of effort produces more collective knowledge. |
| **λ_I** | Private learning efficiency | How much one unit of effort improves the precision of the agent's own signal about θ_{i,t}. The total task precision becomes Y = σ⁻² + λ_I·e + τ_A. | (0, ∞), slider: 0.1–5.0 | Higher λ_I → greater private return to effort → agents work harder → more public signal produced as a side effect → higher X̄_h. |
""")

    # ── Production and welfare ────────────────────────────────────────────────
    with st.expander("⚙️  Production, output, and welfare  —  G(τ), F(X), W"):
        st.markdown("""
| Symbol | Name | Definition | Range | Notes |
|---|---|---|---|---|
| **G(τ)** | Knowledge quality function | G(τ) = 2Φ(√τ) − 1, where Φ is the standard normal CDF. This is the probability that an agent with posterior precision τ correctly identifies the sign of the true state (a ±1 task). G(0) = 0 (no knowledge), G(∞) = 1 (perfect knowledge). It is **concave** and **bounded**. | [0, 1] | The concavity of G is what makes effort a public good: the social marginal product of effort (through X) is higher than agents internalise. |
| **g(τ) = G′(τ)** | Marginal knowledge quality | g(τ) = φ(√τ)/√τ, the derivative of G. Appears in the first-order condition for effort. It is strictly **decreasing** in τ — diminishing returns to precision. | (0, ∞) | — |
| **Output** | Per-agent production | Output = G(X_t) · G(Y_{i,t}) · Δ_X. This is a **Leontief complement**: if either dimension is zero, output is zero regardless of the other. Δ_X is the fixed task-gap parameter (normalised to 1 in the code). | [0, 1] | The Leontief structure is the core assumption: shared and task-specific knowledge are complements, not substitutes. This is why AI — which only improves Y — can destroy welfare by eroding X. |
| **F(X_t)** | Law of motion | X_{t+1} = F(X_t) = [(X_t + λ_G·N·ē*(X_t))⁻¹ + Σ²]⁻¹. Encodes: (1) current knowledge X_t + new public signal λ_G·N·ē* → posterior precision; (2) drift Σ² erodes this into the next period's prior. Fixed points of F are equilibria. | [0, 1/Σ²] | The slope F′ at a fixed point determines stability: F′ < 1 → stable; F′ > 1 → unstable (tipping point). |
| **W** | Social welfare | W = G(X̄_h) · G(Ȳ) − ē*^α/α, evaluated at the high steady state. Increases in τ_A raise G(Ȳ) (AI helps tasks) but lower G(X̄_h) (knowledge erodes) and lower the cost term. The net effect is **hump-shaped**: there is an interior welfare-maximising τ_A* ≤ τ_A^c. | ℝ | — |
""")

    # ── Equilibria ────────────────────────────────────────────────────────────
    with st.expander("⚖️  Equilibria and thresholds  —  X̄_h, X̄_m, τ_A^c, N"):
        st.markdown("""
| Symbol | Name | Definition | Regime | Significance |
|---|---|---|---|---|
| **X̄_h** | High-knowledge equilibrium | The largest positive fixed point of F. Locally stable: if X_0 > X̄_m (the tipping point), the system converges here. **Decreases monotonically** as τ_A rises — AI gradually erodes shared knowledge even before collapse. | Both regimes | The "good" outcome. Society's long-run knowledge stock if it started with adequate initial conditions. |
| **X̄_m** | Tipping point (unstable equilibrium) | The middle fixed point of F, where F′(X̄_m) > 1. It is the **basin boundary**: starting above X̄_m → converge to X̄_h; starting below → collapse to 0. It is very small for small τ_A and grows as τ_A → τ_A^c, at which point X̄_m = X̄_h (saddle-node bifurcation). | Elastic regime only (ε > 4) | The "danger threshold." In practice it is often below grid resolution for small τ_A — meaning the collapse basin is negligible and society is safe — but grows dramatically as AI capability approaches τ_A^c. |
| **X = 0** | Collapse equilibrium | The fixed point at zero knowledge. In the elastic regime (ε > 4) this is **locally stable**: once knowledge falls below X̄_m, the system converges to zero and stays there. In the inelastic regime (ε ≤ 4) it is unstable — knowledge recovers from any positive starting point. | Stable only when ε > 4 | The "bad" outcome. Society loses all shared knowledge permanently. |
| **τ_A^c** | Collapse threshold | The critical AI capability level at which X̄_m and X̄_h collide (saddle-node bifurcation) and the only remaining fixed point is X = 0. Above τ_A^c, **no positive equilibrium exists** and collapse is inevitable regardless of initial conditions. τ_A^c grows **logarithmically in N** (Proposition 6). | Elastic regime only | The key policy boundary. The optimal policy keeps τ_A ≤ τ_A* < τ_A^c. |
| **N** | Community size | Number of agents on the island. Determines the **aggregate public signal precision** λ_G·N·ē*. Larger N → more collective learning → higher X̄_h → larger τ_A^c. The log scaling (τ_A^c ~ ln N) means you need exponentially larger communities to withstand linearly better AI. | 1 to ∞ | Illustrates the limits of the "aggregate" solution: you can delay collapse by building larger communities, but the protection is only logarithmic. |
""")

    # ── Regime ────────────────────────────────────────────────────────────────
    with st.expander("🔀  Regime classification  —  ε = 1/(α−1)"):
        st.markdown("""
The **effort elasticity ε = 1/(α−1)** determines everything qualitative about the model.
It measures how strongly equilibrium effort responds to a change in the value of private learning.

| Elasticity | α range | Behaviour near X = 0 | Steady states | Collapse possible? |
|---|---|---|---|---|
| **ε ≤ 4**  (inelastic) | α ≥ 1.25 | F(X) > X for all small X → X = 0 is **unstable** | Unique positive stable SS X̄_h | **No** — knowledge always recovers |
| **ε > 4**  (elastic) | α < 1.25 | F(X) < X near X = 0 → X = 0 is **stable** | Three SS: 0 (stable), X̄_m (unstable), X̄_h (stable) | **Yes** — path-dependent; depends on initial conditions and τ_A |

**Why ε = 4?**
Near X = 0, effort scales as e*(X) ~ X^{1/(2(α−1))} = X^{ε/2}.
For the collapse SS to be stable, the public signal λ_G·N·e* must grow **slower** than X² as X → 0,
i.e., ε/2 > 2, i.e., **ε > 4**.
If effort falls fast enough (agents stop investing as knowledge gets low),
the self-reinforcing spiral of declining knowledge → declining effort → declining knowledge
leads all the way to zero.
""")

    # ── Public good externality ───────────────────────────────────────────────
    with st.expander("🤝  The public-good externality"):
        st.markdown("""
**The central market failure in the model.**

Each agent's effort e_{i,t} produces two goods simultaneously:
1. **Private**: a signal about their own θ_{i,t} with precision λ_I·e (they keep this).
2. **Public**: a signal about the common θ_t with precision λ_G·e (this goes into the aggregate).

The **aggregate** public signal has precision λ_G·**N**·ē — N times any individual's contribution.
But since there is a continuum of agents (or N is large), each individual's share of this collective good is 1/N → 0.
Agents therefore **ignore the public-good value** of their effort when optimising.

This means equilibrium effort is **socially suboptimal**: agents under-invest in learning relative to the social optimum.
The result is that X̄_h is lower than it would be under a social planner, and the collapse threshold τ_A^c is lower than it could be.

**In the Agent Beliefs tab**, this externality is made visible:
- The **purple ellipse** (individual 1σ) shows each agent's uncertainty using only their own signal.
- The **teal ellipse** (collective 1σ) shows the community's aggregate uncertainty.
- The gap between them is the **externality gap** — the knowledge that exists collectively but that no individual fully internalises.
""")

    # ── Optimal policy ────────────────────────────────────────────────────────
    with st.expander("🏛️  Optimal policy  —  τ_A*, garbling"):
        st.markdown("""
**What should a regulator do?**

The paper (Proposition 8) characterises the optimal policy as **garbling the AI signal**:
reduce the precision agents observe from τ_A to some regulated level τ̃_A ≤ τ_A.

The optimal policy has two phases:
1. **If τ_A ≤ τ_A*** (AI is below welfare-maximising level): **allow full τ_A** — the welfare gains from better task knowledge outweigh the knowledge-erosion cost.
2. **If τ_A > τ_A*** (AI exceeds welfare-maximising level): **cap at τ_A*** — suppress AI precision to prevent further erosion, even if the raw AI signal is better.

**τ_A*** is the peak of the welfare curve W(τ_A). In this explorer, it is shown as the red dashed vertical line in the Social Welfare tab.

Key insight: the optimal policy **never allows τ_A > τ_A^c** (no collapse), and the welfare-maximising τ_A* is strictly less than τ_A^c in most parameterisations. The policy prescription is therefore: monitor AI capability, and if it approaches τ_A*, regulate — not ban — AI by limiting its precision to τ_A*.
""")

    # ── Quick reference ───────────────────────────────────────────────────────
    with st.expander("🗂️  Quick-reference card  —  all symbols at a glance"):
        st.markdown("""
| Symbol | Plain English | Type |
|---|---|---|
| θ_t | True shared knowledge state | Exogenous state (random walk) |
| θ_{i,t} | True task-specific state for agent i | Exogenous state (i.i.d. each period) |
| X_t | Shared knowledge **precision** (= 1/variance of belief about θ_t) | Endogenous state variable |
| Y_{i,t} | Task-knowledge precision (σ⁻² + λ_I·e + τ_A) | Derived from effort and AI |
| e_{i,t} / ē* | Individual effort / equilibrium effort | Choice variable |
| α | Effort cost steepness | Parameter |
| ε = 1/(α−1) | Effort elasticity | Derived parameter (regime switch at ε = 4) |
| λ_G | Public learning efficiency | Parameter |
| λ_I | Private learning efficiency | Parameter |
| Σ² | Knowledge decay rate (drift variance) | Parameter |
| σ⁻² | Task prior precision | Parameter |
| τ_A | AI capability (signal precision) | Parameter / policy variable |
| G(τ) = 2Φ(√τ)−1 | Knowledge quality (bounded [0,1]) | Model function |
| g(τ) = G′(τ) | Marginal knowledge quality | Model function |
| F(X) | Law of motion for X_t | Model function |
| X̄_h | High-knowledge equilibrium (stable) | Endogenous equilibrium |
| X̄_m | Tipping point (unstable equilibrium) | Endogenous equilibrium |
| τ_A^c | Collapse threshold | Endogenous threshold |
| τ_A* | Welfare-maximising AI level | Policy optimum |
| W | Social welfare at high equilibrium | Welfare measure |
| N | Community size (island size) | Parameter |
""")


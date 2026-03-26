"""
Knowledge Collapse — Cross-Disciplinary Recombination Extension
Extension of Acemoglu, Kong & Ozdaglar (2026)

Three knowledge layers:
  Task knowledge    (Y_{i,t})  per-agent, private, resets each period  [unchanged from baseline]
  Domain knowledge  (X_{d,t})  shared within domain, persistent        [original X_t reinterpreted]
  General knowledge (ΣX_d)     sum across domains, analyst metric only  [new — not optimised by agents]

New mechanism:
  AI (τ_A) substitutes for effort → e* falls → public signal weakens → domain knowledge X_d erodes.
  When ΣX_d/k < δ·X_h0, average domain quality reaches the spawn break-even point — agents create d'.
  d' inherits knowledge δ·(Σ w_d·X_d + Σ_{i<j} w_i·X_i·w_j·X_j), where w_d = N_d/N, and agents reallocate via
  credit equilibrium N_d* ∝ G(X_d). Movers carry expertise into d' as a one-time renewal signal.
  General knowledge = ΣX_d may survive and grow even as individual domains collapse.
  When d' itself erodes enough to again depress ΣG below G_h0, d'' is created — cascade continues.

Predictions:
  P1  Recombination reverses AI-induced collapse when δ is large enough — there is a threshold δ*
      above which ΣX_d grows and below which it decays.
  P2  ΣX_d trends clearly up or down; it is almost never flat. One force dominates.
  P3  AI + recomb always outperforms AI + closed (δ > 0 always helps vs δ = 0).
  P4  AI + recomb outperforms no-AI + recomb for all τ_A > τ_c (when δ > δ*).
      AI activates the cascade; without AI there is no collapse and hence no spawning.
      ΣX_d is hump-shaped in τ_A — peaks at intermediate τ_A, slightly declines at very high τ_A.
  P5  ΣX_d exceeds the pre-AI benchmark X_h0 only when δ > δ* (recomb effect > decay effect).

Run with:  streamlit run knowledge_collapse_recomb.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import streamlit as st
from scipy.stats import norm as _norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

# Domain colour palette (up to 8 domains)
_DOMAIN_PALETTE = ["#2563EB", "#DC2626", "#D97706", "#7C3AED",
                   "#059669", "#DB2777", "#0891B2", "#65A30D"]
C_GEN = "#059669"
C_CLO = "#374151"
C_AI  = "#D97706"
C_MIG = "#7C3AED"

# ─────────────────────────────────────────────────────────────────────────────
# Baseline model — unchanged
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
        e_hi *= 10.0
    return float(brentq(_foc, 1e-15, e_hi, args=args, xtol=1e-14, maxiter=300))

def F_step(X, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq):
    e     = solve_effort(X, tau_A, alpha, lambda_I, lambda_G, sigma_inv2)
    inner = X + lambda_G * N * e
    return 1.0 / (1.0 / max(inner, 1e-15) + Sigma_sq)

def find_steady_states(tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq, n_grid=500):
    X_max  = 0.98 / Sigma_sq
    X_log  = np.logspace(-9, np.log10(0.5), n_grid // 2)
    X_lin  = np.linspace(0.5, X_max, n_grid // 2)
    X_grid = np.unique(np.concatenate([X_log, X_lin]))
    kw_    = dict(alpha=alpha, lambda_I=lambda_I, lambda_G=lambda_G,
                  sigma_inv2=sigma_inv2, N=N, Sigma_sq=Sigma_sq)
    resid  = np.array([F_step(x, tau_A, **kw_) - x for x in X_grid])
    ss     = [0.0]
    for i in range(len(resid) - 1):
        if resid[i] * resid[i + 1] < 0:
            try:
                root = brentq(lambda x: F_step(x, tau_A, **kw_) - x,
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

# ─────────────────────────────────────────────────────────────────────────────
# Extension simulation — cascading multi-domain
# ─────────────────────────────────────────────────────────────────────────────

def simulate_open(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq,
                  delta, gamma=0.0):
    """
    Open economy with endogenous domain creation via credit incentives.

    Spawn condition (endogenous — tied to δ, no separate threshold parameter):
      A new domain d' is created whenever average domain knowledge falls to δ × X_h0:
          ΣX_d / k  <  δ · X_h0
      Interpretation: the same δ that governs knowledge transferability also governs
      WHEN it is worth creating a new domain. When existing domains average δ·X_h0,
      the new domain (inheriting δ·ΣX_d = k·δ²·X_h0) matches current quality —
      exactly the break-even point. This fires early (rich inheritance, before collapse)
      and guarantees that each spawn meaningfully contributes to general knowledge.

    At most one new domain per period (cascade is staged, not instantaneous).

    Mechanisms at each spawn:
      - Convergent inheritance:        X_{d',0} = δ·(Σ w_d·X_d + Σ_{i<j} w_i·X_i·w_j·X_j)
                                       where w_d = N_d/N (mover weights); base + recombination bonus
      - Credit-equilibrium allocation: N_d* ∝ G(X_d) for all domains including d'
      - Renewal (move channel):        λ_G · N_{d'} · Σ_d[G(X_d)/Σ_old_G · e_d]
      - Renewal (interact channel):    γ · Σ_{i<j} X_i · X_j  (pairwise synergy)

    General knowledge = ΣX_d (analyst metric; agents optimise only within their domain).
    """
    kw_ = dict(alpha=alpha, lambda_I=lambda_I, lambda_G=lambda_G,
               sigma_inv2=sigma_inv2, N=N, Sigma_sq=Sigma_sq)

    ss0  = find_steady_states(0.0, **kw_)
    X_h0 = max((s for s in ss0 if s > 0.05), default=1.0)
    G_h0 = float(G(X_h0))          # pre-AI credit benchmark: ΣG must stay above this

    # Each domain: {"X", "X_init", "N", "renewal_next"}
    domains = [{"X": X_h0, "X_init": X_h0, "N": float(N), "renewal_next": 0.0}]

    X_paths = {0: np.zeros(T)}
    N_paths = {0: np.zeros(T)}
    gen_times = []

    for t in range(T):
        # ── 1. Record X (before update) ──────────────────────────────────────
        for i in range(len(domains)):
            X_paths[i][t] = domains[i]["X"]

        # ── 2. Endogenous spawn check ─────────────────────────────────────────
        # Spawn whenever average domain X < δ·X_h0.
        # Fires every period once collapse is underway — that is intentional.
        # Each spawn adds δ·ΣX_d to total; with strong δ the cascade sustains
        # and grows ΣX_d indefinitely. With weak δ each spawn is too thin to
        # overcome decay and ΣX_d still falls.
        n_before = len(domains)
        sum_X    = sum(domains[j]["X"] for j in range(n_before))

        if delta > 0.0 and sum_X / n_before < delta * X_h0:

            efforts = [solve_effort(domains[j]["X"], tau_A, alpha,
                                    lambda_I, lambda_G, sigma_inv2)
                       for j in range(n_before)]

            # ── Convergent inheritance (mover-weighted + interaction) ────────
            # w_d = fraction of agents in domain d — determines what d contributes
            # base     = weighted avg of what movers carry from their source domains
            # interact = pairwise product of contributions — the recombination bonus
            #            (meeting of diverse knowledges creates extra value)
            # X_new    = δ · (base + interact)   [NOT a function of raw ΣX_d]
            N_total_before = sum(domains[j]["N"] for j in range(n_before))
            if N_total_before < 1e-15:
                w = [1.0 / n_before] * n_before
            else:
                w = [domains[j]["N"] / N_total_before for j in range(n_before)]
            contribs = [w[j] * domains[j]["X"] for j in range(n_before)]
            base     = sum(contribs)
            interact = sum(contribs[i] * contribs[j]
                           for i in range(n_before)
                           for j in range(i + 1, n_before))
            X_new = delta * (base + interact)

            # ── Credit-equilibrium agent allocation: N_d* ∝ G(X_d) ───────────
            G_existing = [float(G(domains[j]["X"])) for j in range(n_before)]
            G_new      = float(G(X_new))
            G_all      = G_existing + [G_new]
            total_G    = sum(G_all)
            if total_G < 1e-15:
                N_alloc = [float(N) / (n_before + 1)] * (n_before + 1)
            else:
                N_alloc = [float(N) * gv / total_G for gv in G_all]
            N_new = N_alloc[-1]

            # ── Renewal: move channel ─────────────────────────────────────────
            old_total_G = sum(G_existing)
            if old_total_G < 1e-15:
                wt = [1.0 / n_before] * n_before
            else:
                wt = [G_existing[j] / old_total_G for j in range(n_before)]
            renewal_move = lambda_G * N_new * sum(
                wt[j] * efforts[j] for j in range(n_before))

            # ── Renewal: pairwise cross-domain complementarity ────────────────
            renewal_interact = gamma * sum(
                domains[j]["X"] * domains[k]["X"]
                for j in range(n_before) for k in range(j + 1, n_before))

            # ── Reallocate agents to equilibrium ──────────────────────────────
            for j in range(n_before):
                domains[j]["N"] = N_alloc[j]

            child = {"X": X_new, "X_init": X_new, "N": N_new,
                     "renewal_next": renewal_move + renewal_interact}
            ci = len(domains)
            domains.append(child)
            X_paths[ci] = np.full(T, np.nan)
            N_paths[ci] = np.full(T, np.nan)
            X_paths[ci][t] = X_new
            gen_times.append(t)

        # ── 3. Record N AFTER spawning ────────────────────────────────────────
        for i in range(len(domains)):
            N_paths[i][t] = domains[i]["N"]

        # ── 4. Update X for all domains ───────────────────────────────────────
        for idx, dom in enumerate(domains):
            e      = solve_effort(dom["X"], tau_A, alpha, lambda_I, lambda_G, sigma_inv2)
            inner  = dom["X"] + lambda_G * dom["N"] * e + dom["renewal_next"]
            dom["X"]           = 1.0 / (1.0 / max(inner, 1e-15) + Sigma_sq)
            dom["renewal_next"] = 0.0

    n_dom = len(domains)
    all_X = np.array([X_paths[i] for i in range(n_dom)])
    all_N = np.array([N_paths[i] for i in range(n_dom)])
    X_general = np.nansum(all_X, axis=0)

    # Peak excludes t=0 (which is always X_h0 by initialisation and not a reversal event)
    _peak_arr = X_general[1:] if T > 1 else X_general
    _peak_val = float(np.nanmax(_peak_arr))
    _peak_t   = int(1 + int(np.nanargmax(_peak_arr))) if T > 1 else 0

    return dict(
        domain_X=all_X, domain_N=all_N,
        X_general=X_general,
        X_general_peak=_peak_val,   # highest ΣX_d reached after t=0
        X_general_peak_t=_peak_t,
        gen_times=gen_times,
        n_domains=n_dom,
        X_h0=X_h0, G_h0=G_h0, T=T,
    )


def simulate_closed(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq):
    """Closed economy (delta=0): single domain, original law of motion."""
    kw_ = dict(alpha=alpha, lambda_I=lambda_I, lambda_G=lambda_G,
               sigma_inv2=sigma_inv2, N=N, Sigma_sq=Sigma_sq)
    ss0  = find_steady_states(0.0, **kw_)
    X    = max((s for s in ss0 if s > 0.05), default=1.0)
    path = np.zeros(T)
    for t in range(T):
        path[t] = X
        X = F_step(X, tau_A, **kw_)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Knowledge Collapse — Recombination Extension", layout="wide")
st.title("Knowledge Collapse — Recombination Extension")
st.caption("Extension of Acemoglu, Kong & Ozdaglar (2026) · "
           "Cross-disciplinary recombination as a second spillover channel")

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = dict(tau_A=1.0,  delta=0.5,  gamma=0.0)
# Strong-reversal config: τ_A above τ_c → original domain collapses → spawn fires.
# δ=0.85: rich inheritance — new domain gets 85% of ΣX_d at each spawn.
# γ=0.05: pairwise synergy bonus at spawn — combining two knowledge traditions
#          produces more than the sum of parts (Weitzman recombination).
_STRONG   = dict(tau_A=0.85, delta=0.85, gamma=0.05)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.page_link("pages/home.py", label="🏠 Back to Home", icon="🏠")
    st.divider()
    st.header("Parameters")

    st.subheader("Baseline")
    alpha      = st.slider("Cost steepness  α",           1.05, 2.0,  1.20, 0.01)
    eps = 1.0 / (alpha - 1.0)
    if eps > 4:
        st.error(f"ε = {eps:.2f}  →  Elastic · collapse possible")
    else:
        st.success(f"ε = {eps:.2f}  →  Inelastic · unique stable equilibrium")
    N          = st.slider("Community size  N",            5, 500,  50,  5)
    lambda_G   = st.slider("Public learning  λ_G",        0.1, 5.0, 1.0, 0.1)
    lambda_I   = st.slider("Private learning  λ_I",       0.1, 5.0, 1.0, 0.1)
    Sigma_sq   = st.slider("Knowledge decay  Σ²",         0.005, 0.2, 0.05, 0.005)
    sigma_inv2 = st.slider("Task prior precision  σ⁻²",   0.001, 2.0, 0.01, 0.001)
    tau_A      = st.slider("AI capability  τ_A",          0.0, 3.0, step=0.01, key="tau_A")

    st.divider()
    st.subheader("Recombination Extension")

    def _load_strong():
        for k, v in _STRONG.items():
            st.session_state[k] = v

    st.button("⚡ Load strong-reversal config",
              on_click=_load_strong,
              help="τ_A=0.85 (above τ_c → collapse triggers spawning), δ=0.85, γ=0.05.  "
                   "Rich inheritance (85% of ΣX_d) at each spawn, plus pairwise synergy bonus (γ) "
                   "from combining existing knowledge traditions.  "
                   "Result: ΣX_d is sustained above X_h0 — P1 reversal demonstrated.")

    st.caption(
        "**Spawn trigger (credit-equilibrium):** a new domain is created when ΣX_d/k < δ·X_h0. "
        "At this point the new domain inherits δ·(average X) ≈ δ²·X_h0 per existing domain — "
        "exactly matching current quality. This is the *break-even* condition from the credit "
        "equilibrium: agents enter d' when its expected credit return equals the return in existing domains. "
        "The same δ governs both *how much* knowledge transfers and *when* entry is worthwhile."
    )

    delta = st.slider(
        "Knowledge transferability  δ", 0.0, 1.0, step=0.01, key="delta",
        help="Fraction of ΣX_d inherited by new domain at spawn. "
             "δ=0 → closed economy (baseline recovered exactly).")
    gamma = st.slider(
        "Cross-domain complementarity  γ", 0.0, 0.3, step=0.005, key="gamma",
        help="Weitzman recombination: combining two knowledge traditions produces more than either alone. "
             "Economics + psychology → behavioural economics. Physics + chemistry → molecular biology. "
             "Formally: γ·Σ_{i<j} X_i·X_j added to the renewal signal at each spawn — "
             "richer predecessors produce larger boosts, growing superlinearly in domain count. "
             "γ=0 → purely additive (movers carry knowledge but no cross-tradition synergy).")

    st.divider()
    T = st.slider("Time horizon  T", 50, 500, 200, 10)

kw = dict(alpha=alpha, lambda_I=lambda_I, lambda_G=lambda_G,
          sigma_inv2=sigma_inv2, N=N, Sigma_sq=Sigma_sq)

@st.cache_data(max_entries=64)
def cached_tau_c(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq):
    return find_collapse_threshold(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq)

@st.cache_data(max_entries=128)
def cached_open(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq,
                delta, gamma):
    return simulate_open(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2,
                         N, Sigma_sq, delta, gamma)

@st.cache_data(max_entries=128)
def cached_closed(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq):
    return simulate_closed(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq)

@st.cache_data(max_entries=32)
def cached_tau_sweep(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq,
                     delta, gamma, T, n_pts=12):
    T_sw   = min(T, 60)   # sweeps only need enough time for cascade to develop
    tau_c  = find_collapse_threshold(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq)
    taus   = np.linspace(0.0, min(tau_c * 2.0, 3.0), n_pts)
    x_open, x_clos = [], []
    for t_A in taus:
        sim_o = simulate_open(T_sw, t_A, alpha, lambda_I, lambda_G, sigma_inv2,
                              N, Sigma_sq, delta, gamma)
        sim_c = simulate_closed(T_sw, t_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq)
        x_open.append(sim_o["X_general"][-1])
        x_clos.append(sim_c[-1])
    return taus, np.array(x_open), np.array(x_clos), tau_c

@st.cache_data(max_entries=32)
def cached_delta_sweep(tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq,
                       gamma, T, n_pts=12):
    T_sw   = min(T, 60)
    deltas  = np.linspace(0.0, 1.0, n_pts)
    x_final = []
    for d in deltas:
        sim = simulate_open(T_sw, tau_A, alpha, lambda_I, lambda_G, sigma_inv2,
                            N, Sigma_sq, d, gamma)
        x_final.append(sim["X_general"][-1])
    return deltas, np.array(x_final)

tau_c   = cached_tau_c(**kw)
sim_o   = cached_open(T, tau_A, **kw, delta=delta, gamma=gamma)
sim_c   = cached_closed(T, tau_A, **kw)
sim_noai = cached_open(T, 0.0,   **kw, delta=delta, gamma=gamma)

n_dom     = sim_o["n_domains"]
gen_times = sim_o["gen_times"]
X_h0      = sim_o["X_h0"]
G_h0      = sim_o["G_h0"]

# ── Top metrics ───────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Collapse threshold  τ_A^c", f"{tau_c:.3f}")
m2.metric("AI capability  τ_A",        f"{tau_A:.3f}",
          delta="above threshold" if tau_A >= tau_c else "below threshold")
m3.metric("Transferability  δ",        f"{delta:.2f}")
m4.metric("Domains spawned",           str(n_dom - 1))
_open_T    = sim_o['X_general'][-1]
_closed_T  = sim_c[-1]
_X_h0_top  = sim_o['X_h0']
_ratio_top = _open_T / _X_h0_top
_vs_pre_ai = "above pre-AI ✅" if _open_T >= _X_h0_top else "below pre-AI"
_vs_closed = f"{_open_T / max(_closed_T, 1e-8):.1f}× closed"
m5.metric("ΣX_d(T) / X_h0  (reversal)",
          f"{_ratio_top:.2f}×",
          delta=f"{_vs_pre_ai}  |  {_vs_closed}",
          delta_color="normal",
          help="PRIMARY METRIC: ΣX_d(T)/X_h0 ≥ 1.0 means total knowledge across all domains "
               "exceeds the pre-AI single-domain steady state — genuine reversal of Acemoglu's result. "
               "Below 1.0 means degradation, even if open >> closed.")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌐 Domain Dynamics",
    "📊 Four-Way Comparison",
    "🔬 P3+P4 — Role of AI",
    "📏 P1+P5 — Recomb threshold",
    "📈 P2 — Knowledge trends",
    "📖 Extension Guide",
])

# ── helpers ───────────────────────────────────────────────────────────────────
def domain_color(i):
    return _DOMAIN_PALETTE[i % len(_DOMAIN_PALETTE)]

def domain_label(i):
    labels = ["d₀ (original)", "d₁", "d₂", "d₃", "d₄", "d₅", "d₆", "d₇"]
    return labels[i] if i < len(labels) else f"d{i}"

# ─── Tab 1: Domain Dynamics ────────────────────────────────────────────────────
with tab1:
    ts = np.arange(T)
    st.caption(
        "Per-domain knowledge X_d (coloured lines) and total general knowledge ΣX_d (green dashed).  "
        "Gold horizontal line = no-AI counterfactual X_h0 — what knowledge would be if AI was never introduced.  "
        "Strong reversal = ΣX_d climbs above the gold line.  "
        "Vertical markers = endogenous spawn events (spawn when ΣX_d/k < δ·X_h0)."
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    MAX_PLOT = 8  # max individual domain lines shown; rest aggregated
    n_shown  = min(n_dom, MAX_PLOT)

    ax = axes[0]
    for i in range(n_shown):
        mask = ~np.isnan(sim_o["domain_X"][i])
        ax.plot(ts[mask], sim_o["domain_X"][i][mask],
                color=domain_color(i), lw=1.8, alpha=0.85, label=domain_label(i))
    if n_dom > MAX_PLOT:
        ax.plot([], [], color="grey", lw=1.2, alpha=0.5,
                label=f"+{n_dom - MAX_PLOT} more domains (in ΣX_d)")
    ax.plot(ts, sim_o["X_general"], color=C_GEN, lw=2.5, ls="--",
            label="General knowledge  ΣX_d")
    ax.axhline(X_h0, color="goldenrod", lw=2.0, ls="-.",
               label=f"No-AI counterfactual  X_h0 = {X_h0:.2f}")
    # show only first few spawn markers to avoid clutter
    shown_gt = gen_times[:20]
    for gt in shown_gt:
        ax.axvline(gt, color=C_MIG, lw=1.0, ls="--", alpha=0.4)
    if gen_times:
        ax.axvline(gen_times[0], color=C_MIG, lw=1.0, ls="--", alpha=0.4,
                   label=f"Spawn events ({len(gen_times)} total)")
    ax.set_xlabel("Period  t", fontsize=11)
    ax.set_ylabel("Knowledge precision  X", fontsize=11)
    ax.set_title(f"Domain knowledge paths  ·  τ_A={tau_A:.2f}  δ={delta:.2f}  γ={gamma:.3f}", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    # Stackplot: show first MAX_PLOT domains, aggregate the rest
    N_full  = np.nan_to_num(sim_o["domain_N"], nan=0.0)
    N_stack = N_full[:n_shown]
    labels_n = [domain_label(i) for i in range(n_shown)]
    colors_n = [domain_color(i) for i in range(n_shown)]
    if n_dom > MAX_PLOT:
        N_rest = N_full[MAX_PLOT:].sum(axis=0)
        N_stack = np.vstack([N_stack, N_rest])
        labels_n.append(f"+{n_dom - MAX_PLOT} other domains")
        colors_n.append("lightgrey")
    ax2.stackplot(ts, *N_stack, labels=labels_n, colors=colors_n, alpha=0.65)
    for gt in shown_gt:
        ax2.axvline(gt, color=C_MIG, lw=1.2, ls="--", alpha=0.45)
    ax2.set_xlabel("Period  t", fontsize=11)
    ax2.set_ylabel("Number of agents", fontsize=11)
    ax2.set_title("Agent distribution across domains", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylim(0, N * 1.05)
    ax2.grid(True, alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    # ── Key outcome metrics ───────────────────────────────────────────────────
    open_final     = sim_o["X_general"][-1]
    open_peak      = sim_o["X_general_peak"]
    open_peak_t    = sim_o["X_general_peak_t"]
    closed_final   = sim_c[-1]
    peak_ratio     = open_peak  / X_h0   # THE primary metric
    final_ratio    = open_final / X_h0
    open_vs_closed = open_final / max(closed_final, 1e-8)
    n_spawns       = len(gen_times)

    st.markdown("#### Key question: does ΣX_d(T) stay above the pre-AI baseline X_h0?")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("X_h0  (pre-AI baseline)", f"{X_h0:.3f}",
              help="No-AI single-domain steady state — the benchmark for reversal of Acemoglu's result.")
    m2.metric("Final  ΣX_d / X_h0  ★",
              f"{final_ratio:.2f}×",
              delta=f"{'✅ above X_h0 — strong reversal' if open_final > X_h0 else '❌ below X_h0 — load ⚡ config'}",
              delta_color="normal",
              help="★ PRIMARY METRIC.  ΣX_d at end of simulation T.  "
                   "> 1.0× = sustained reversal: recombination keeps knowledge above pre-AI level.")
    m3.metric("Peak  ΣX_d / X_h0",
              f"{peak_ratio:.2f}×",
              delta=f"at t={open_peak_t}",
              delta_color="normal",
              help="Highest ΣX_d reached (after t=0).  Shows the maximum reversal achieved.")
    m4.metric("Open / Closed at T",
              f"{open_vs_closed:.1f}×",
              help="Secondary metric: recombination preserved how much more knowledge than the no-recombination counterfactual.")

    st.caption(
        f"**Strong prediction (vs pre-AI):** ΣX_d(T)/X_h0 = **{final_ratio:.2f}** at T={sim_o['T']}"
        f"{'  ✅  Sustained above pre-AI level' if open_final > X_h0 else '  ❌  Did not sustain X_h0 — load ⚡ strong-reversal config'}"
        f"   ·   **Weak prediction (vs closed):** Open/Closed = {open_vs_closed:.1f}×"
    )

    # Per-domain breakdown (compact)
    dcols = st.columns(min(n_dom, 6))
    for i in range(min(n_dom, 6)):
        last = sim_o["domain_X"][i]
        final = last[~np.isnan(last)][-1] if (~np.isnan(last)).any() else 0.0
        dcols[i].metric(f"X_d{domain_label(i)}", f"{final:.3f}")

    # ── Regime badge ─────────────────────────────────────────────────────────
    # Strong reversal = ΣX_d(T) > X_h0 (sustained, not just a transient spike)
    if final_ratio > 1.0:
        regime_label, regime_fn = "✅ Strong reversal — ΣX_d(T) > X_h0 sustained (Acemoglu reversed)", st.success
    elif open_vs_closed > 5.0 and tau_A > tau_c:
        regime_label, regime_fn = "⚠️  Collapse prevented vs closed, but ΣX_d never reached X_h0", st.warning
    elif open_vs_closed > 2.0:
        regime_label, regime_fn = "⚠️  Moderate buffer — open > closed, ΣX_d < X_h0", st.warning
    else:
        regime_label, regime_fn = "❌  Weak buffer — try ⚡ strong-reversal config", st.error
    regime_fn(
        f"**{regime_label}**   |   "
        f"Peak ΣX_d/X_h0 = **{peak_ratio:.2f}×** (t={open_peak_t})   "
        f"Final ΣX_d/X_h0 = {final_ratio:.2f}×   "
        f"Open/Closed = {open_vs_closed:.1f}×   "
        f"Spawns = {n_spawns}  ·  γ = {gamma:.3f}"
    )

# ─── Tab 2: Four-Way Comparison ───────────────────────────────────────────────
with tab2:
    st.caption(
        "**Four scenarios — reading order = causal logic of the model:**  "
        "**Gold** — initial general knowledge X_h0 (pre-AI, single domain).  "
        "**Grey** — no-AI + open: recombination infrastructure exists but no AI pressure; knowledge stays flat at X_h0 (AI is the driver, not openness alone).  "
        "**Teal** — AI + open: AI forces specialisation, recombination recycles knowledge; peak ΣX_d can exceed X_h0 (*strong reversal*).  "
        "**Red** — AI + closed (Acemoglu): AI forces specialisation, no recombination; knowledge permanently collapses.  "
        "The gap between teal and gold is the **net gain from AI + recombination**. The gap between teal and red is the **value of openness**."
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: level paths ─────────────────────────────────────────────────────
    ax = axes[0]
    # Gold: initial general knowledge / pre-AI baseline
    ax.axhline(X_h0, color="goldenrod", lw=2.2, ls="-.",
               label=f"Initial general knowledge  X_h0 = {X_h0:.2f}")
    # Grey: no-AI + open (τ_A=0 with open economy)
    ax.plot(ts, sim_noai["X_general"], color="grey", lw=1.8, ls=":",
            label="No-AI + open  (τ_A=0)")
    # Red: AI + closed
    ax.plot(ts, sim_c, color=C_CLO, lw=2.0, ls="--",
            label=f"AI + closed  (Acemoglu,  δ=0,  τ_A={tau_A:.2f})")
    # Teal: AI + open — plot last so it sits on top
    ax.plot(ts, sim_o["X_general"], color=C_GEN, lw=2.5,
            label=f"AI + open  (this paper,  δ={delta:.2f},  τ_A={tau_A:.2f})")
    # Spawn markers
    for gt in gen_times:
        ax.axvline(gt, color=C_MIG, lw=1.0, ls=":", alpha=0.4)
    if gen_times:
        ax.axvline(gen_times[0], color=C_MIG, lw=1.0, ls=":", alpha=0.4,
                   label="Domain spawn events")
    ax.set_xlabel("Period  t", fontsize=11)
    ax.set_ylabel("Knowledge precision  ΣX_d", fontsize=11)
    ax.set_title("General knowledge: four-way comparison", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ── Right: ratio to X_h0 (normalised view) ───────────────────────────────
    ax2 = axes[1]
    ax2.axhline(1.0, color="goldenrod", lw=2.2, ls="-.",
                label="X_h0 baseline = 1.0  (strong-reversal threshold)")
    ax2.plot(ts, sim_noai["X_general"] / X_h0, color="grey", lw=1.8, ls=":",
             label="No-AI + open")
    ax2.plot(ts, sim_c / X_h0, color=C_CLO, lw=2.0, ls="--",
             label="AI + closed")
    ax2.plot(ts, sim_o["X_general"] / X_h0, color=C_GEN, lw=2.5,
             label="AI + open")
    for gt in gen_times:
        ax2.axvline(gt, color=C_MIG, lw=1.0, ls=":", alpha=0.4)
    ax2.set_xlabel("Period  t", fontsize=11)
    ax2.set_ylabel("ΣX_d / X_h0", fontsize=11)
    ax2.set_title("Knowledge normalised to pre-AI baseline", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Summary metrics ───────────────────────────────────────────────────────
    _peak_ratio  = sim_o["X_general_peak"] / X_h0
    _final_ratio = sim_o["X_general"][-1]  / X_h0
    _vs_closed   = sim_o["X_general"][-1]  / max(sim_c[-1], 1e-8)
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak ΣX_d / X_h0",
              f"{_peak_ratio:.2f}×",
              delta="Strong reversal ✅" if _peak_ratio >= 1.0 else "Below pre-AI")
    c2.metric("Final ΣX_d / X_h0",
              f"{_final_ratio:.2f}×",
              delta="Above pre-AI ✅" if _final_ratio >= 1.0 else "Below pre-AI")
    c3.metric("AI+open / AI+closed  (value of openness)",
              f"{_vs_closed:.1f}×")

# ─── Tab 3: P3 + P4 ───────────────────────────────────────────────────────────
with tab3:
    st.caption(
        "**P3:** AI+recomb always outperforms AI+closed (recombination is always beneficial).  "
        "**P4:** AI+recomb outperforms no-AI+recomb — AI *helps* by triggering the cascade. "
        "Higher τ_A → more frequent spawning → higher ΣX_d."
    )
    run_p34 = st.checkbox("Run τ_A sweep (~20 simulations)", key="run_p34")
    if run_p34:
        taus_sw, x_open_sw, x_clos_sw, tc_sw = cached_tau_sweep(
            **kw, delta=delta, gamma=gamma, T=T)
        # X_h0 for normalisation: no-AI baseline (closed, τ_A=0)
        _noai_level = float(simulate_closed(T, 0.0, **kw)[-1])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # ── Left: P3 — open always above closed ──────────────────────────────
        ax = axes[0]
        ax.plot(taus_sw, x_open_sw, color=C_GEN, lw=2.2,
                label=f"AI + open  ΣX_d(T)  (δ={delta:.2f})")
        ax.plot(taus_sw, x_clos_sw, color=C_CLO, lw=2.0, ls="--",
                label="AI + closed  X_d(T)  (δ=0, Acemoglu)")
        ax.fill_between(taus_sw, x_clos_sw, x_open_sw,
                        where=x_open_sw >= x_clos_sw,
                        alpha=0.15, color=C_GEN, label="Recomb advantage")
        ax.axvline(tc_sw, color=C_AI, lw=1.5, ls=":",
                   label=f"τ_A^c = {tc_sw:.3f}")
        ax.axvline(tau_A, color=C_MIG, lw=1.3, ls="--", alpha=0.6,
                   label=f"Current τ_A = {tau_A:.2f}")
        ax.set_xlabel("AI capability  τ_A", fontsize=11)
        ax.set_ylabel(f"General knowledge  ΣX_d(T)", fontsize=11)
        ax.set_title("P3: AI+recomb always beats AI+closed", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # ── Right: P4 — AI helps; ΣX_d(open) rises with τ_A ─────────────────
        ax2 = axes[1]
        ax2.plot(taus_sw, x_open_sw, color=C_GEN, lw=2.2,
                 label=f"AI + open  (δ={delta:.2f})")
        ax2.axhline(_noai_level, color="grey", lw=1.8, ls=":",
                    label=f"No-AI + open  ≈ {_noai_level:.3f}  (τ_A=0, no cascade)")
        ax2.axvline(tc_sw, color=C_AI, lw=1.5, ls=":", alpha=0.7,
                    label=f"τ_A^c = {tc_sw:.3f}")
        ax2.fill_between(taus_sw, _noai_level, x_open_sw,
                         where=x_open_sw >= _noai_level,
                         alpha=0.15, color=C_GEN, label="AI+recomb > no-AI")
        ax2.set_xlabel("AI capability  τ_A", fontsize=11)
        ax2.set_ylabel(f"ΣX_d(T)  (AI+open)", fontsize=11)
        ax2.set_title("P4: AI enables cascade — AI+recomb > no-AI for all τ_A > τ_A^c", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        above_tc = taus_sw > tc_sw
        if above_tc.any():
            p3_holds = (x_open_sw[above_tc] >= x_clos_sw[above_tc]).all()
            # P4: AI+recomb > no-AI for all τ_A > τ_c (the cascade jump, not monotone slope)
            p4_holds = (x_open_sw[above_tc] >= _noai_level).all()
            p4_min   = x_open_sw[above_tc].min() / max(_noai_level, 1e-8)
            if p3_holds:
                st.success("P3 confirmed: AI+recomb ≥ AI+closed for all τ_A > τ_A^c.")
            else:
                st.warning("P3 violated for some τ_A > τ_A^c — check δ.")
            if p4_holds:
                st.success(f"P4 confirmed: AI+recomb > no-AI for all τ_A > τ_A^c "
                           f"(min ratio = {p4_min:.2f}×). Crossing τ_A^c activates the cascade — "
                           f"AI is a complement to recombination.")
            else:
                st.warning("P4 not confirmed: AI+recomb dips below no-AI baseline for some τ_A. "
                           "Try higher δ above the recombination threshold.")
        else:
            st.info("Raise τ_A or lower α to see above-threshold behaviour.")
    else:
        st.info("Check the box above to run the sweep. Results are cached after first run.")

# ─── Tab 4: P1 + P5 ───────────────────────────────────────────────────────────
with tab4:
    st.caption(
        "**P1:** Recombination reverses collapse when δ is large enough — there is a threshold δ* "
        "above which ΣX_d grows and below which it decays.  "
        "**P5:** AI+recomb exceeds the original X_h0 benchmark *only* when recomb > decay (δ > δ*)."
    )
    run_p15 = st.checkbox("Run δ sweep (~15 simulations)", key="run_p15")
    if run_p15:
        tau_A_run = tau_A if tau_A > tau_c else tau_c * 1.15
        deltas_sw, x_delta_sw = cached_delta_sweep(tau_A_run, **kw, gamma=gamma, T=T)
        _x_h0_ref = float(simulate_closed(T, 0.0, **kw)[-1])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # ── Left: P1+P5 — ΣX_d(T) vs δ with X_h0 threshold line ─────────────
        ax = axes[0]
        ax.plot(deltas_sw, x_delta_sw, color=C_GEN, lw=2.5,
                label=f"AI + open  ΣX_d(T)  (τ_A={tau_A_run:.2f})")
        ax.axhline(_x_h0_ref, color="goldenrod", lw=2.0, ls="-.",
                   label=f"Original X_h0 = {_x_h0_ref:.3f}  (no-AI, no-recomb)")
        ax.axvline(delta, color=C_MIG, lw=1.3, ls=":", alpha=0.7,
                   label=f"Current δ = {delta:.2f}")
        # Mark threshold δ*
        above = x_delta_sw >= _x_h0_ref
        if above.any() and not above.all():
            _delta_star_idx = np.argmax(above)
            _delta_star = deltas_sw[_delta_star_idx]
            ax.axvline(_delta_star, color="red", lw=1.5, ls="--",
                       label=f"δ* ≈ {_delta_star:.2f}  (reversal threshold)")
        ax.fill_between(deltas_sw, _x_h0_ref, x_delta_sw,
                        where=x_delta_sw >= _x_h0_ref,
                        alpha=0.15, color=C_GEN, label="Reversal zone (P1+P5)")
        ax.fill_between(deltas_sw, x_delta_sw, _x_h0_ref,
                        where=x_delta_sw < _x_h0_ref,
                        alpha=0.10, color=C_CLO, label="Still below X_h0")
        ax.set_xlabel("Transferability  δ", fontsize=11)
        ax.set_ylabel(f"ΣX_d(T)", fontsize=11)
        ax.set_title("P1+P5: Recombination threshold δ*", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # ── Right: same, normalised to X_h0, at two τ_A levels ───────────────
        tau_A_hi = min(tau_A_run * 1.5 + 0.3, 3.0)
        deltas_sw2, x_delta_sw2 = cached_delta_sweep(tau_A_hi, **kw, gamma=gamma, T=T)

        ax2 = axes[1]
        ax2.plot(deltas_sw,  x_delta_sw  / _x_h0_ref, color=C_GEN, lw=2.2,
                 label=f"τ_A = {tau_A_run:.2f}")
        ax2.plot(deltas_sw2, x_delta_sw2 / _x_h0_ref, color=C_AI,  lw=2.0, ls="--",
                 label=f"τ_A = {tau_A_hi:.2f}  (higher AI)")
        ax2.axhline(1.0, color="goldenrod", lw=2.0, ls="-.",
                    label="X_h0 = 1.0")
        ax2.axvline(delta, color=C_MIG, lw=1.3, ls=":", alpha=0.7,
                    label=f"Current δ = {delta:.2f}")
        ax2.set_xlabel("Transferability  δ", fontsize=11)
        ax2.set_ylabel("ΣX_d(T) / X_h0", fontsize=11)
        ax2.set_title("Normalised: both τ_A levels cross 1.0 at δ*", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if above.any() and not above.all():
            st.success(f"P1+P5 confirmed: reversal threshold δ* ≈ {_delta_star:.2f}. "
                       f"For δ > δ*, ΣX_d(T) > X_h0 (recomb beats decay). "
                       f"For δ < δ*, ΣX_d(T) < X_h0 (decay wins).")
        elif above.all():
            st.success("ΣX_d(T) > X_h0 for all δ — recomb dominates at this τ_A. "
                       "Try lower τ_A to find threshold.")
        else:
            st.warning("ΣX_d(T) < X_h0 for all δ — raise τ_A above τ_A^c or check parameters.")
    else:
        st.info("Check the box above to run the sweep. Results are cached after first run.")

# ─── Tab 5: P2 — Knowledge trends ────────────────────────────────────────────
with tab5:
    st.caption(
        "**P2:** If recombination > decay, ΣX_d trends upward. If decay > recombination, "
        "ΣX_d trends downward. General knowledge is almost never flat — it is either "
        "accumulating or collapsing."
    )
    run_p2 = st.checkbox("Run trend simulations (3 simulations)", key="run_p2")
    if run_p2:
        tau_A_run2 = tau_A if tau_A > tau_c else tau_c * 1.15
        # Three δ values: low (decay wins), mid (near threshold), high (recomb wins)
        d_low  = max(delta * 0.3, 0.05)
        d_mid  = delta
        d_high = min(delta * 1.5 + 0.2, 0.98)

        sim_low  = simulate_open(T, tau_A_run2, **kw, delta=d_low,  gamma=gamma)
        sim_mid  = simulate_open(T, tau_A_run2, **kw, delta=d_mid,  gamma=gamma)
        sim_high = simulate_open(T, tau_A_run2, **kw, delta=d_high, gamma=gamma)
        _x_h0_p2 = sim_low["X_h0"]
        ts2 = np.arange(T)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # ── Left: raw ΣX_d trajectories ──────────────────────────────────────
        ax = axes[0]
        ax.plot(ts2, sim_low["X_general"],  color="#DC2626", lw=2.0,
                label=f"δ={d_low:.2f}  (decay wins)")
        ax.plot(ts2, sim_mid["X_general"],  color=C_AI,      lw=2.0, ls="--",
                label=f"δ={d_mid:.2f}  (current)")
        ax.plot(ts2, sim_high["X_general"], color=C_GEN,     lw=2.2,
                label=f"δ={d_high:.2f}  (recomb wins)")
        ax.axhline(_x_h0_p2, color="goldenrod", lw=1.8, ls="-.",
                   label=f"X_h0 = {_x_h0_p2:.3f}")
        ax.set_xlabel("Period  t", fontsize=11)
        ax.set_ylabel("General knowledge  ΣX_d", fontsize=11)
        ax.set_title(f"P2: Knowledge trends  (τ_A={tau_A_run2:.2f})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # ── Right: late-period trend (last 40% of T) to show slope ───────────
        t_start = int(T * 0.6)
        ax2 = axes[1]
        for sim, col, lbl in [
            (sim_low,  "#DC2626", f"δ={d_low:.2f}"),
            (sim_mid,  C_AI,      f"δ={d_mid:.2f}"),
            (sim_high, C_GEN,     f"δ={d_high:.2f}"),
        ]:
            yy = sim["X_general"][t_start:]
            xx = ts2[t_start:]
            ax2.plot(xx, yy, color=col, lw=2.0, label=lbl)
            # Fit linear trend over late period
            slope, intercept = np.polyfit(xx, yy, 1)
            ax2.plot(xx, slope * xx + intercept, color=col, lw=1.2, ls=":",
                     alpha=0.7, label=f"  trend: {slope:+.4f}/period")
        ax2.axhline(_x_h0_p2, color="goldenrod", lw=1.5, ls="-.", alpha=0.7)
        ax2.set_xlabel("Period  t  (late phase)", fontsize=11)
        ax2.set_ylabel("ΣX_d", fontsize=11)
        ax2.set_title("Late-period trend: slope shows direction", fontsize=12)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Slope diagnostics
        results = []
        for sim, d_val in [(sim_low, d_low), (sim_mid, d_mid), (sim_high, d_high)]:
            yy = sim["X_general"][t_start:]
            xx = np.arange(len(yy), dtype=float)
            slope = np.polyfit(xx, yy, 1)[0]
            results.append((d_val, slope))
        cols = st.columns(3)
        for col, (d_val, slope) in zip(cols, results):
            direction = "↑ Growing" if slope > 1e-5 else ("↓ Declining" if slope < -1e-5 else "→ Flat")
            col.metric(f"δ = {d_val:.2f}", direction, f"slope = {slope:+.5f}/period")
    else:
        st.info("Check the box above to run the simulations.")

# ─── Tab 6: Extension Guide ───────────────────────────────────────────────────
with tab6:
    st.markdown("## Extension Guide")
    st.markdown(
        "Complete design rationale, formal definitions, and economic intuition for every "
        "modelling choice in the extension. Read sequentially — each section builds on the last."
    )
    st.divider()

    with st.expander("🏗️  Architecture: why three knowledge layers?", expanded=True):
        st.markdown(f"""
The original paper has one knowledge layer: **X_t** — the shared knowledge precision of one
community on one question.  We extend to three layers with distinct economic roles.

---
**Layer 1 — Task knowledge  Y_{{i,t}}   (per-agent, private, resets every period)**

Precision:  Y_{{i,t}} = σ⁻² + λ_I·e_{{i,t}} + τ_A

Represents an agent's knowledge of their *idiosyncratic* problem θ_{{i,t}}.
Each period θ_{{i,t}} ~ N(0, 1) is drawn fresh — it is the agent's specific context
(a patient's symptoms, a client's brief, a particular engineering challenge).
Because the context changes every period, yesterday's task knowledge is useless today.
The AI signal τ_A enters here, acting as a free private signal that substitutes for effort.
**Unchanged from the baseline paper.**

---
**Layer 2 — Domain knowledge  X_{{d,t}}   (shared within one domain, persistent)**

Follows the original law of motion:  X_{{d,t+1}} = [(X_{{d,t}} + λ_G·N_d·ē*(X_{{d,t}}))⁻¹ + Σ²]⁻¹

The original paper's X_t is reinterpreted as domain-specific: one island = one research domain.
This knowledge is the accumulated theoretical framework that agents in a field share —
it improves when agents exert effort that produces public signals (λ_G·N_d·ē), and it decays
slowly each period (drift Σ²).  Unlike task knowledge, it persists and compounds over time.
**The baseline model is recovered exactly by setting δ = 0 (no new domains ever form).**

---
**Layer 3 — General knowledge  ΣX_d   (Bayesian sum, analyst metric only)**

ΣX_d = Σ_d X_{{d,t}}   (current values:  {sim_o['X_general'][-1]:.3f},  benchmark X_h0 = {X_h0:.3f})

Precisions sum because domains provide *independent signals* about related epistemic targets.
A Bayesian agent with access to information from k independent domains with precisions
X_1, …, X_k has combined precision Σ X_d.

**Why agents do NOT optimise over ΣX_d — a critical design choice:**
Agents observe only their own domain's X_{{d,t}}.  They do not know how many other domains exist,
what their precision levels are, or that a global aggregate even exists.
This is realistic: a researcher in macroeconomics optimises given the state of macroeconomic knowledge,
not the total stock of all human knowledge.  If agents could observe and optimise over ΣX_d,
their FOC would change fundamentally (they would internalise the value of multi-domain knowledge),
which would be a different model entirely.  ΣX_d is the *analyst's* measure of societal knowledge
after aggregating across domains — it is not an object in any agent's information set.
""")

    with st.expander("⚙️  Two spillover channels — and why AI activates the second"):
        st.markdown("""
The model has two routes by which effort translates into knowledge.

---
**Channel 1 — Within-domain refinement   (baseline)**

e → private signal precision λ_I·e → improves Y_{i,t}  (task knowledge)
e → public signal precision λ_G·e → aggregates to λ_G·N_d·ē → improves X_{d,t+1}

This is the original paper's mechanism.  Each unit of effort simultaneously helps the agent
learn about their own task AND contributes to the domain's collective knowledge.
Because the aggregate matters N times as much as any individual's contribution,
agents dramatically under-invest in this public good.

**AI weakens Channel 1:**  τ_A raises Y_{i,t} directly, so agents reduce effort ē*.
Lower effort → weaker public signal → X_{d,t} erodes → eventually collapses.

---
**Channel 2 — Cross-domain renewal   (extension)**

Domain erosion → trigger fires → new domain d' spawns
Agents migrate from all existing domains to d'
Movers arrive with expertise (encoded as effort-weighted signals) applied in the new context
→ one-time renewal boost to X_{d',0}

**AI indirectly activates Channel 2:**  by eroding X_{d,t}, AI causes the trigger to fire,
which spawns a new domain and provides a fresh start.  The same force that destroys
Channel 1 activates Channel 2 — this makes recombination a **self-correcting response**
to AI-induced collapse.

In a closed economy, only Channel 1 exists and collapse is **absorbing** (once X → 0, recovery
requires an exogenous shock).  In an open economy, Channel 2 provides a second route
that AI cannot directly suppress.
""")

    # pre-compute for display inside expander
    _final_Xs  = [sim_o["domain_X"][i][~np.isnan(sim_o["domain_X"][i])][-1]
                  for i in range(n_dom) if (~np.isnan(sim_o["domain_X"][i])).any()]
    sum_X_final = sum(_final_Xs) / max(len(_final_Xs), 1)

    with st.expander("⚠️  Design decision: endogenous credit-incentive trigger (not exogenous threshold)"):
        st.markdown(f"""
**Why not an exogenous threshold?**

Early versions fired a spawn when any domain fell below some fixed fraction of its initial value
(e.g., `trigger_frac × X_{{d,0}}`).  This is exogenous: the threshold is set by the modeller,
not derived from any agent's incentive.  There are two problems:

1. **No economic foundation.** Why 30%? Why 60%? The number has no connection to the credit
   mechanism that drives everything else in the model.

2. **Immediate cascade bug.** If the threshold is global (fraction of X_h0), children are born
   below the threshold and spawn immediately in the same period — all domains appear at t = 0.

**The correct approach: use δ itself as both the inheritance rate and the spawn threshold.**

The key insight: the same δ that governs *how much* knowledge transfers to a new domain should
also govern *when* it is worth creating one.  There is no separate trigger parameter.

**Endogenous spawn condition:**
>  **ΣX_d / k  <  δ · X_h0**
>
>  (Average domain knowledge has fallen to δ fraction of the pre-AI benchmark.)

Interpretation: when existing domains average at δ·X_h0, a new domain would be born with
X_{{d',0}} = δ·ΣX_d = k·δ²·X_h0.  That is exactly **δ·(current average)** — the new domain
matches the current standard of the existing domains.  This is the break-even point: the new
domain is as productive as the average existing one, so creating it is immediately worthwhile.

**Why this gives the right timing:**

- Fires **early** (when domains are at δ·X_h0, still meaningfully productive).  With δ = 0.85,
  spawns fire when domains have declined by only 15% from X_h0 — inheritance is rich.
- The higher δ is, the earlier spawning fires and the richer the inheritance.
  δ is simultaneously "how easily knowledge transfers" and "how soon it is worth transferring."
- At δ = 0: spawn condition requires average X < 0 — never fires.  Closed economy recovered.

**On spawn frequency:**

Once the cascade begins, the trigger fires every period — and this is intentional.
After spawn, the new domain inherits δ·(average X), which is itself below δ·X_h0 (since the
average was just at that threshold). So the next period's average is still below δ·X_h0 and
the trigger fires again. This means new domains are continuously created as old ones decay.

With weak δ: each spawn adds very little, ΣX_d stays near zero — collapse confirmed.
With strong δ: each spawn adds substantially, ΣX_d grows — reversal confirmed.
The frequency is not the mechanism; the inheritance size is.

**Current state:**  X_h0 = {X_h0:.3f},  δ·X_h0 = {delta*X_h0:.3f} (spawn threshold),
current ΣX_d/k = {sum_X_final:.3f} (where sum_X_final = ΣX_d(T)/{n_dom}),
domains spawned = {n_dom - 1}
""")

    with st.expander("🔀  Convergent inheritance — why ALL domains contribute"):
        st.markdown(f"""
**Chain inheritance (wrong):**  when d₁ spawns d₂, only d₁ contributes:
X_{{d₂,0}} = δ · X_{{d₁}}

This misses the fundamental point: when researchers from multiple fields jointly create a new
field, they bring knowledge from ALL their fields, not just the most recent one.

**Convergent inheritance (correct):**  ALL active domains contribute to the new one:
> **X_{{d',0}} = δ · (Σ_d w_d · X_d  +  γ · Σ_{{i<j}} w_i·X_i · w_j·X_j)**

where **w_d = N_d / N** is domain d's agent share (from the credit equilibrium).
The first term is the agent-weighted average knowledge — what movers collectively carry.
The second term (γ > 0) is the pairwise synergy: knowledge traditions interacting, not just pooling.

The parameter δ ∈ [0, 1] is the **knowledge transferability** — how much survives the move
to a new context.  δ = 0 → nothing transfers (closed economy recovered exactly).
δ = 1 → full weighted average transfers.

**Note on inheritance size:**  the weighted average Σ w_d·X_d ≤ max(X_d), so each new domain
starts below the best existing domain.  With δ < 1, it starts below the average too.
The cascade sustains ΣX_d when δ is large enough that the continuous stream of spawns
accumulates faster than decay removes knowledge from existing domains.

Current config: δ = {delta:.2f},  spawns so far = {n_dom - 1},
actual ΣX_d(T)/X_h0 = {sim_o['X_general'][-1]/X_h0:.3f}

**Baseline recovered at δ = 0:**  no knowledge transfers, spawning has no effect,
ΣX_d = X_{{d₀,t}} — identical to the closed economy.
""")

    with st.expander("👥  Credit-equilibrium migration — endogenous, not ad hoc"):
        st.markdown(f"""
An earlier version used δ to govern BOTH knowledge transferability AND the fraction of agents
who migrate.  This was explicitly rejected: **δ is knowledge, not agents**.
Conflating the two makes δ do double duty with no economic justification.

**The right approach: agents migrate until they are indifferent between domains.**

Each agent earns expected credit proportional to **G(X_d)** in domain d.
With N_d agents in domain d, per-agent expected credit = G(X_d) / N_d.
In equilibrium, no agent benefits from switching domains:
> G(X_{{d₁}}) / N_{{d₁}} = G(X_{{d₂}}) / N_{{d₂}} = … = G(X_{{d'}}) / N_{{d'}} = common value

Solving for N_d given fixed total N:
> **N_d* = N · G(X_d) / Σ_j G(X_j)   for all domains including d'**

This is the **credit-equilibrium allocation** — endogenous, derived from the same G(X) credit
function that drives the effort FOC.  The number of movers to d' is determined entirely by
how attractive d' is (its G(X_{{d',0}})) relative to the attractiveness of staying in existing domains.

**Renewal signal (move channel):**  movers from each domain d carry effort e_d weighted by
domain attractiveness G(X_d) / Σ_old_G.  Their expertise enters d' as a one-time renewal boost:
renewal_move = λ_G · N_{{d'}} · Σ_d[ G(X_d)/Σ_old_G · e_d ]

Current equilibrium example:  with {n_dom} domains, agents are distributed as:
{', '.join(f'd{i}: {sim_o["domain_N"][i][min(T-1, np.where(~np.isnan(sim_o["domain_X"][i]))[0][-1] if (~np.isnan(sim_o["domain_X"][i])).any() else 0)]:.1f}' for i in range(min(n_dom, 8)))}{'  (+{} more)'.format(n_dom-8) if n_dom > 8 else ''} agents
""")

    with st.expander("✨  Cross-domain complementarity γ — why additive is not recombination"):
        st.markdown(f"""
Without γ, the renewal signal only reflects expertise movers bring (the "move channel").
This is **additive consolidation** — new domain benefits from predecessor knowledge but
knowledge traditions don't interact.

With γ > 0, **combining two knowledge traditions produces more than the sum of parts**.

**Renewal signal (interact channel):**
renewal_interact = γ · Σ_{{i<j}} X_i · X_j

Each pair of existing domains (i, j) contributes a bonus proportional to both their knowledge
levels.  The intuition (Schumpeterian / Weitzman): combining economics and biology gave
behavioural economics; combining physics and chemistry gave molecular biology.
The insight emerges from the *combination itself*, not from either field alone.
With k domains of equal level X, the total synergy is k(k−1)/2 · γ · X² —
growing quadratically in domain count.  As the cascade lengthens, synergies compound.

**Why this makes the distinction between additive and multiplicative:**
- γ = 0:  ΣX_d at T depends only on δ and spawn count.  Two generations always give (1+δ)².
- γ > 0:  richer predecessors produce larger boosts.  The growth in ΣX_d becomes superlinear
  in domain count — the more fields that existed before, the more the new field benefits from
  their interaction.

γ = {gamma:.3f} currently.  {'No complementarity — purely additive recombination.' if gamma < 0.005 else 'Synergy bonus per spawn ≈ {:.3f}  (last spawn, first 8 domains)'.format(gamma * sum(sim_o["domain_X"][i][np.where(~np.isnan(sim_o["domain_X"][i]))[0][-1]] * sim_o["domain_X"][j][np.where(~np.isnan(sim_o["domain_X"][j]))[0][-1]] for i in range(min(n_dom,8)) for j in range(i+1, min(n_dom,8)) if (~np.isnan(sim_o["domain_X"][i])).any() and (~np.isnan(sim_o["domain_X"][j])).any())) if n_dom > 1 else ''}
""")

    with st.expander("📐  Five predictions — mechanism to test"):
        st.markdown(f"""
**P1 — Recombination reverses collapse when δ is large enough**

There is a threshold δ* above which ΣX_d grows over time and below which it decays.
At δ*, recombination exactly offsets knowledge decay.  The threshold exists because
each spawn injects δ·ΣX_d into general knowledge, while the decay Σ² continuously erodes it.

*Test:*  Tab 4 (P1+P5 — δ sweep). ΣX_d(T) vs δ should show a clear crossing at δ*.

---

**P2 — ΣX_d trends clearly up or down; almost never flat**

The dynamics are driven by the competition between recombination gain (at each spawn)
and continuous knowledge decay (each period).  This competition produces a clear trend —
not a hovering near steady-state.  For δ < δ*, ΣX_d declines monotonically.
For δ > δ*, ΣX_d rises.  The transition at δ* is sharp.

*Test:*  Tab 5 (P2 — Knowledge trends). Trajectories at δ below/above δ* show clear slopes.

---

**P3 — AI+recomb (average, end, peak) always beats AI+closed**

Recombination is always better than no recombination, regardless of δ or τ_A.
Even a tiny δ > 0 generates some inheritance at spawn that the closed economy never gets.
Open/Closed = **{sim_o['X_general'][-1] / max(sim_c[-1], 1e-8):.1f}×** at current parameters.

*Test:*  Tab 3 (P3+P4). Green curve always above dark curve.

---

**P4 — AI+recomb beats no-AI+recomb: AI *helps* knowledge when recombination is possible**

Without AI (τ_A = 0): no collapse → no spawning → ΣX_d stays at X_h0.
With AI (τ_A > τ_c): collapse triggers spawning → recombination cascade → ΣX_d jumps above X_h0.
AI, by destabilising individual domains, activates the cascade that would never fire otherwise.
This makes AI a *complement* to recombination: the open economy needs AI's pressure to generate
the cascade, and the cascade needs the open economy to convert that pressure into knowledge growth.

Note: ΣX_d is hump-shaped in τ_A — it peaks at an intermediate τ_A and slightly declines at very
high τ_A (domains collapse too fast for inheritance to accumulate). P4's claim is that AI+recomb
exceeds no-AI+recomb for *all* τ_A > τ_A^c (the jump at τ_A^c), not that ΣX_d is monotone in τ_A.

*Test:*  Tab 3 (P3+P4, right panel). ΣX_d(open) should lie above no-AI baseline for all τ_A > τ_A^c.

---

**P5 — AI+recomb exceeds X_h0 only when recomb > decay**

ΣX_d(T) > X_h0 (the pre-AI, no-recomb benchmark) if and only if δ > δ*.
Below δ*, the cascade adds some knowledge but decay dominates — ΣX_d stays below X_h0.
Above δ*, recombination accumulates faster than decay removes — full reversal of Acemoglu.

*Test:*  Tab 4 (P1+P5). X_h0 threshold line shows exactly where the crossing occurs.
Current: ΣX_d(T)/X_h0 = {sim_o['X_general'][-1]/X_h0:.2f}×

---
**Quick-reference: load the strong-reversal config (⚡ button)**

δ = 0.85, γ = 0.05, τ_A = 0.85

With endogenous trigger and these parameters:
- τ_A = 0.85 > τ_A^c ({tau_c:.3f}): domains actively erode, continuously triggering spawning.
- δ = 0.85: each new domain inherits 85% of the weighted-average knowledge across existing domains —
  rich inheritance means each spawn contributes meaningfully to ΣX_d.
- γ = 0.05: pairwise Weitzman synergy bonus at each spawn — combining knowledge traditions
  produces more than the weighted sum alone.
- Spawning fires every period once collapse begins — the cascade sustains ΣX_d because
  each spawn adds enough (via δ=0.85) to outpace the continuous decay Σ².
""")

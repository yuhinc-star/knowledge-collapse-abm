"""
Knowledge Collapse — Cross-Disciplinary Recombination Extension
Extension of Acemoglu, Kong & Ozdaglar (2026)

Three knowledge layers:
  Task knowledge    (Y_{i,t})  per-agent, private, resets each period  [unchanged from baseline]
  Domain knowledge  (X_{d,t})  shared within domain, persistent        [original X_t reinterpreted]
  General knowledge (ΣX_d)     sum across domains, analyst metric only  [new — not optimised by agents]

New mechanism:
  AI (τ_A) substitutes for effort → e* falls → public signal weakens → domain knowledge X_d erodes.
  When ΣG(X_d) < G(X_h0), agents' total credit falls below the pre-AI benchmark — they create d'.
  d' inherits knowledge δ·ΣX_d (convergent: ALL domains contribute) and agents reallocate via
  credit equilibrium N_d* ∝ G(X_d). Movers carry expertise into d' as a one-time renewal signal.
  General knowledge = ΣX_d may survive and grow even as individual domains collapse.
  When d' itself erodes enough to again depress ΣG below G_h0, d'' is created — cascade continues.

Predictions:
  P1  Open economy (δ > 0) weakens or reverses AI-induced collapse of general knowledge.
  P2  The benefit of openness is largest when AI capability is strongest.

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
                  delta, gamma=0.0, max_domains=8):
    """
    Open economy with endogenous domain creation via credit incentives.

    Spawn condition (fully endogenous — no exogenous threshold):
      A new domain d' is created whenever total per-agent credit across all existing
      domains falls below the pre-AI benchmark:
          ΣG(X_d) / N  <  G(X_h0) / N   ⟺   ΣG(X_d) < G(X_h0)
      Interpretation: agents collectively create a new domain when AI has degraded
      their credit pool below the no-AI equilibrium level. This is individually
      rational — a new domain with inheritance X_{d',0} = δ·ΣX_d offers positive
      credit that the existing pool can no longer match at the pre-AI standard.

    At most one new domain per period (cascade is staged, not instantaneous).

    Mechanisms at each spawn:
      - Convergent inheritance:        X_{d',0} = δ · ΣX_d  (ALL domains contribute)
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
        # Spawn iff total credit has fallen below pre-AI benchmark AND room remains
        n_before = len(domains)
        sum_G    = sum(float(G(domains[j]["X"])) for j in range(n_before))

        if delta > 0.0 and sum_G < G_h0 and n_before < max_domains:

            efforts = [solve_effort(domains[j]["X"], tau_A, alpha,
                                    lambda_I, lambda_G, sigma_inv2)
                       for j in range(n_before)]

            # ── Convergent inheritance ────────────────────────────────────────
            total_X = sum(domains[j]["X"] for j in range(n_before))
            X_new   = delta * total_X

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
        for dom in domains:
            e     = solve_effort(dom["X"], tau_A, alpha, lambda_I, lambda_G, sigma_inv2)
            inner = dom["X"] + lambda_G * dom["N"] * e + dom["renewal_next"]
            dom["X"]           = 1.0 / (1.0 / max(inner, 1e-15) + Sigma_sq)
            dom["renewal_next"] = 0.0

    n_dom = len(domains)
    all_X = np.array([X_paths[i] for i in range(n_dom)])
    all_N = np.array([N_paths[i] for i in range(n_dom)])
    X_general = np.nansum(all_X, axis=0)

    return dict(
        domain_X=all_X, domain_N=all_N,
        X_general=X_general,
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
_DEFAULTS = dict(tau_A=1.0,  delta=0.5,  gamma=0.0,  max_domains=4)
_STRONG   = dict(tau_A=0.85, delta=0.85, gamma=0.05, max_domains=8)
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
              help="Sets δ=0.85, γ=0.05, τ_A=0.85, max_domains=8 — "
                   "the parameter region where open economy strongly reverses collapse.")

    st.caption(
        "**Spawn trigger (endogenous):** a new domain is created whenever "
        "ΣG(X_d) < G(X_h0) — total agent credit falls below the pre-AI benchmark. "
        "No exogenous threshold needed; timing emerges from credit incentives."
    )

    delta = st.slider(
        "Knowledge transferability  δ", 0.0, 1.0, step=0.01, key="delta",
        help="Fraction of ΣX_d inherited by new domain at spawn. "
             "δ=0 → closed economy (baseline recovered exactly).")
    gamma = st.slider(
        "Cross-domain complementarity  γ", 0.0, 0.3, step=0.005, key="gamma",
        help="Pairwise synergy bonus added to renewal at spawn: γ·Σ_{i<j} X_i·X_j. "
             "Captures the insight that combining two knowledge traditions produces "
             "more than the sum of parts. γ=0 → purely additive recombination.")
    max_domains = st.slider(
        "Max domains", 2, 12, step=1, key="max_domains",
        help="Cap on total number of domains (prevents infinite cascade).")

    st.divider()
    T = st.slider("Time horizon  T", 50, 500, 200, 10)

kw = dict(alpha=alpha, lambda_I=lambda_I, lambda_G=lambda_G,
          sigma_inv2=sigma_inv2, N=N, Sigma_sq=Sigma_sq)

@st.cache_data(max_entries=64)
def cached_tau_c(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq):
    return find_collapse_threshold(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq)

@st.cache_data(max_entries=128)
def cached_open(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq,
                delta, gamma, max_domains):
    return simulate_open(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2,
                         N, Sigma_sq, delta, gamma, max_domains)

@st.cache_data(max_entries=128)
def cached_closed(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq):
    return simulate_closed(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq)

@st.cache_data(max_entries=32)
def cached_tau_sweep(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq,
                     delta, gamma, max_domains, T, n_pts=20):
    tau_c  = find_collapse_threshold(alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq)
    taus   = np.linspace(0.0, min(tau_c * 2.0, 3.0), n_pts)
    x_open, x_clos = [], []
    for t_A in taus:
        sim_o = simulate_open(T, t_A, alpha, lambda_I, lambda_G, sigma_inv2,
                              N, Sigma_sq, delta, gamma, max_domains)
        sim_c = simulate_closed(T, t_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq)
        x_open.append(sim_o["X_general"][-1])
        x_clos.append(sim_c[-1])
    return taus, np.array(x_open), np.array(x_clos), tau_c

@st.cache_data(max_entries=32)
def cached_delta_sweep(tau_A, alpha, lambda_I, lambda_G, sigma_inv2, N, Sigma_sq,
                       gamma, max_domains, T, n_pts=15):
    deltas  = np.linspace(0.0, 1.0, n_pts)
    x_final = []
    for d in deltas:
        sim = simulate_open(T, tau_A, alpha, lambda_I, lambda_G, sigma_inv2,
                            N, Sigma_sq, d, gamma, max_domains)
        x_final.append(sim["X_general"][-1])
    return deltas, np.array(x_final)

tau_c = cached_tau_c(**kw)
sim_o = cached_open(T, tau_A, **kw, delta=delta, gamma=gamma, max_domains=max_domains)
sim_c = cached_closed(T, tau_A, **kw)

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
m5.metric("General knowledge at T",    f"{sim_o['X_general'][-1]:.3f}",
          delta=f"{sim_o['X_general'][-1] - sim_c[-1]:+.3f} vs closed")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Domain Dynamics",
    "📊 Open vs Closed",
    "🔬 P1 — τ_A sweep",
    "📏 P2 — δ sweep",
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
        "Per-domain knowledge X_d (one line per domain) and general knowledge ΣX_d (green dashed). "
        "New domains spawn endogenously when total agent credit ΣG(X_d) falls below pre-AI benchmark G(X_h0). "
        "Dashed horizontal = X_h0 (benchmark for strong reversal). Dotted = closed economy baseline."
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for i in range(n_dom):
        mask = ~np.isnan(sim_o["domain_X"][i])
        ax.plot(ts[mask], sim_o["domain_X"][i][mask],
                color=domain_color(i), lw=2.0, label=domain_label(i))
    ax.plot(ts, sim_o["X_general"], color=C_GEN, lw=2.2, ls="--",
            label="General knowledge  ΣX_d")
    ax.plot(ts, sim_c, color=C_CLO, lw=1.5, ls=":",
            label="Closed economy  X_d₀  (δ=0)")
    ax.axhline(X_h0, color=C_AI, lw=1.1, ls=":", alpha=0.55,
               label=f"Pre-AI benchmark  X_h0 = {X_h0:.2f}  (strong reversal: ΣX_d exceeds this)")
    for gt in gen_times:
        ax.axvline(gt, color=C_MIG, lw=1.2, ls="--", alpha=0.45)
    if gen_times:
        ax.axvline(gen_times[0], color=C_MIG, lw=1.2, ls="--", alpha=0.45,
                   label="Domain spawn events")
    ax.set_xlabel("Period  t", fontsize=11)
    ax.set_ylabel("Knowledge precision", fontsize=11)
    ax.set_title(f"Domain knowledge paths  ·  τ_A = {tau_A:.3f}  ·  δ = {delta:.2f}", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    # Build stacked N arrays, filling NaN with 0 for stackplot
    N_stack = np.nan_to_num(sim_o["domain_N"], nan=0.0)
    labels_n  = [domain_label(i) for i in range(n_dom)]
    colors_n  = [domain_color(i) for i in range(n_dom)]
    ax2.stackplot(ts, *N_stack, labels=labels_n, colors=colors_n, alpha=0.65)
    for gt in gen_times:
        ax2.axvline(gt, color=C_MIG, lw=1.2, ls="--", alpha=0.45)
    ax2.set_xlabel("Period  t", fontsize=11)
    ax2.set_ylabel("Number of agents", fontsize=11)
    ax2.set_title("Agent distribution across domains", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylim(0, N * 1.05)
    ax2.grid(True, alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    cols = st.columns(min(n_dom + 2, 6))
    cols[0].metric("X_h0 (initial)", f"{X_h0:.3f}")
    cols[1].metric("ΣX_d at T", f"{sim_o['X_general'][-1]:.3f}")
    for i in range(min(n_dom, len(cols) - 2)):
        last = sim_o["domain_X"][i]
        final = last[~np.isnan(last)][-1] if (~np.isnan(last)).any() else 0.0
        cols[i + 2].metric(f"X_{domain_label(i)} at T", f"{final:.3f}")

    # ── Tipping-point badge ───────────────────────────────────────────────────
    # With convergent inheritance, ΣX_d grows by (1+δ) at each spawn event.
    # Condition for ΣX_d_final > X_h0: need (1+δ)^k > decay, k = spawns.
    # We read the actual outcome from the simulation.
    reversal_ratio = sim_o["X_general"][-1] / X_h0
    just_above     = tau_c < tau_A < tau_c + 0.4
    if reversal_ratio > 0.70 and tau_A > tau_c:
        regime_label, regime_fn = "Strong reversal", st.success
    elif reversal_ratio > 0.30 and tau_A > tau_c:
        regime_label, regime_fn = "Moderate buffer", st.warning
    else:
        regime_label, regime_fn = "Weak buffer / no collapse", st.error
    n_spawns = len(gen_times)
    regime_fn(
        f"**Regime: {regime_label}**  ·  ΣX_d(T)/X_h0 = {reversal_ratio:.2f}"
        f"  ·  (1+δ)^spawns = {(1+delta)**n_spawns:.2f}  ·  γ = {gamma:.3f}"
    )

# ─── Tab 2: Open vs Closed ────────────────────────────────────────────────────
with tab2:
    st.caption(
        "General knowledge in the open economy (ΣX_d, green) vs domain knowledge "
        "in the closed economy (X_d₀, dark). Bottom panel: period-over-period change "
        "in ΣX_d — shows when general knowledge is growing, stable, or still declining."
    )

    # Pre-compute rate of change
    dX_gen  = np.diff(sim_o["X_general"], prepend=sim_o["X_general"][0])
    dX_clo  = np.diff(sim_c,              prepend=sim_c[0])
    gap     = sim_o["X_general"] - sim_c

    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [1.1, 1]})

    # ── Row 1: level + benefit gap ────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(ts, sim_o["X_general"], color=C_GEN, lw=2.2,
            label=f"Open economy  ΣX_d  (δ={delta:.2f})")
    ax.plot(ts, sim_c,              color=C_CLO, lw=2.0, ls="--",
            label="Closed economy  X_d  (δ=0)")
    for gt in gen_times:
        ax.axvline(gt, color=C_MIG, lw=1.2, ls=":", alpha=0.5)
    if gen_times:
        ax.axvline(gen_times[0], color=C_MIG, lw=1.2, ls=":", alpha=0.5,
                   label="Domain spawn events")
    ax.set_xlabel("Period  t", fontsize=11)
    ax.set_ylabel("Knowledge precision", fontsize=11)
    ax.set_title("General knowledge level: open vs closed", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    ax2 = axes[0, 1]
    ax2.fill_between(ts, 0, gap, where=gap >= 0, alpha=0.3, color=C_GEN,
                     label="Open > Closed")
    ax2.fill_between(ts, 0, gap, where=gap <  0, alpha=0.3, color=C_CLO,
                     label="Closed > Open")
    ax2.plot(ts, gap, color=C_GEN, lw=1.8)
    ax2.axhline(0, color="black", lw=0.8, alpha=0.4)
    for gt in gen_times:
        ax2.axvline(gt, color=C_MIG, lw=1.2, ls=":", alpha=0.5)
    ax2.set_xlabel("Period  t", fontsize=11)
    ax2.set_ylabel("ΣX_d(open) − X_d(closed)", fontsize=11)
    ax2.set_title("Recombination benefit over time", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # ── Row 2: rate of change ΔΣX_d/t ────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.plot(ts, dX_gen, color=C_GEN, lw=2.0,
             label=f"ΔΣX_d  open  (δ={delta:.2f})")
    ax3.plot(ts, dX_clo, color=C_CLO, lw=1.8, ls="--",
             label="ΔX_d  closed  (δ=0)")
    ax3.axhline(0, color="black", lw=1.0, alpha=0.5)
    ax3.fill_between(ts, 0, dX_gen, where=dX_gen >= 0,
                     alpha=0.25, color=C_GEN, label="Growing")
    ax3.fill_between(ts, 0, dX_gen, where=dX_gen <  0,
                     alpha=0.20, color=C_AI,  label="Declining")
    for gt in gen_times:
        ax3.axvline(gt, color=C_MIG, lw=1.2, ls=":", alpha=0.5)
    if gen_times:
        ax3.axvline(gen_times[0], color=C_MIG, lw=1.2, ls=":", alpha=0.5,
                    label="Spawn events")
    ax3.set_xlabel("Period  t", fontsize=11)
    ax3.set_ylabel("ΔΣX_d / period", fontsize=11)
    ax3.set_title("Rate of change of general knowledge  ΔΣX_d", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2)

    # ── Cumulative gain ───────────────────────────────────────────────────────
    ax4 = axes[1, 1]
    cum_gain = np.cumsum(dX_gen - dX_clo)
    ax4.plot(ts, cum_gain, color=C_GEN, lw=2.0,
             label="Cumulative extra knowledge (open − closed)")
    ax4.fill_between(ts, 0, cum_gain, where=cum_gain >= 0,
                     alpha=0.2, color=C_GEN)
    ax4.fill_between(ts, 0, cum_gain, where=cum_gain <  0,
                     alpha=0.2, color=C_CLO)
    ax4.axhline(0, color="black", lw=0.8, alpha=0.4)
    for gt in gen_times:
        ax4.axvline(gt, color=C_MIG, lw=1.2, ls=":", alpha=0.5)
    ax4.set_xlabel("Period  t", fontsize=11)
    ax4.set_ylabel("Σ(ΔΣX_d − ΔX_d_closed)", fontsize=11)
    ax4.set_title("Cumulative extra knowledge from recombination", fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ─── Tab 3: P1 ────────────────────────────────────────────────────────────────
with tab3:
    st.caption(
        "**Prediction P1:** Open economy (δ > 0) weakens or reverses AI-induced collapse. "
        "Green curve (open ΣX_d) should stay above dark curve (closed) past τ_A^c."
    )
    run_p1 = st.checkbox("Run τ_A sweep (~20 simulations)", key="run_p1")
    if run_p1:
        taus_sw, x_open_sw, x_clos_sw, tc_sw = cached_tau_sweep(
            **kw, delta=delta, gamma=gamma, max_domains=max_domains, T=T)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        ax.plot(taus_sw, x_open_sw, color=C_GEN, lw=2.2,
                label=f"Open economy  ΣX_d(T)  (δ={delta:.2f})")
        ax.plot(taus_sw, x_clos_sw, color=C_CLO, lw=2.0, ls="--",
                label="Closed economy  X_d(T)  (δ=0)")
        ax.axvline(tc_sw, color=C_AI, lw=1.5, ls=":",
                   label=f"Collapse threshold  τ_A^c = {tc_sw:.3f}")
        ax.axvline(tau_A, color=C_MIG, lw=1.3, ls="--", alpha=0.6,
                   label=f"Current τ_A = {tau_A:.3f}")
        ax.set_xlabel("AI capability  τ_A", fontsize=11)
        ax.set_ylabel(f"General knowledge at t = {T}", fontsize=11)
        ax.set_title("P1: Open economy weakens AI-induced collapse", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        ax2 = axes[1]
        benefit = x_open_sw - x_clos_sw
        ax2.plot(taus_sw, benefit, color=C_GEN, lw=2.2)
        ax2.fill_between(taus_sw, 0, benefit, where=benefit >= 0,
                         alpha=0.25, color=C_GEN, label="Open > Closed")
        ax2.fill_between(taus_sw, 0, benefit, where=benefit <  0,
                         alpha=0.25, color=C_CLO, label="Open < Closed")
        ax2.axhline(0, color="black", lw=0.8, alpha=0.4)
        ax2.axvline(tc_sw, color=C_AI, lw=1.5, ls=":", alpha=0.6,
                    label=f"τ_A^c = {tc_sw:.3f}")
        ax2.set_xlabel("AI capability  τ_A", fontsize=11)
        ax2.set_ylabel("ΣX_d(open) − X_d(closed)  at t = T", fontsize=11)
        ax2.set_title("Recombination benefit vs AI capability", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

        above_tc = taus_sw > tc_sw
        if above_tc.any():
            frac_positive = (benefit[above_tc] > 0).mean()
            if frac_positive > 0.6:
                st.success(f"P1 supported: open economy outperforms closed in "
                           f"{frac_positive:.0%} of τ_A > τ_A^c cases.")
            else:
                st.warning(f"P1 mixed: open economy outperforms in only {frac_positive:.0%} "
                           f"of τ_A > τ_A^c cases. Try higher δ or lower trigger threshold.")
        else:
            st.info("Raise τ_A or lower α to see above-threshold behavior.")
    else:
        st.info("Check the box above to run the sweep. Results are cached after first run.")

# ─── Tab 4: P2 ────────────────────────────────────────────────────────────────
with tab4:
    st.caption(
        "**Prediction P2:** The benefit of openness is largest when AI is strongest. "
        "Shows how general knowledge at time T rises with δ, compared across two τ_A levels."
    )
    run_p2 = st.checkbox("Run δ sweep (~30 simulations)", key="run_p2")
    if run_p2:
        deltas_sw,  x_delta_sw  = cached_delta_sweep(tau_A,    **kw, gamma=gamma,
                                                      max_domains=max_domains, T=T)
        tau_A_hi = min(tau_A * 1.5 + 0.3, 3.0)
        deltas_sw2, x_delta_sw2 = cached_delta_sweep(tau_A_hi, **kw, gamma=gamma,
                                                      max_domains=max_domains, T=T)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        ax.plot(deltas_sw,  x_delta_sw,  color=C_GEN, lw=2.2,
                label=f"τ_A = {tau_A:.3f}  (current)")
        ax.plot(deltas_sw2, x_delta_sw2, color=C_AI,  lw=2.0, ls="--",
                label=f"τ_A = {tau_A_hi:.3f}  (higher AI)")
        ax.axvline(delta, color=C_MIG, lw=1.3, ls=":", alpha=0.7,
                   label=f"Current δ = {delta:.2f}")
        ax.set_xlabel("Transferability  δ", fontsize=11)
        ax.set_ylabel(f"General knowledge  ΣX_d(T)", fontsize=11)
        ax.set_title("P2: General knowledge vs transferability", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        ax2 = axes[1]
        b_low = x_delta_sw  - x_delta_sw[0]
        b_hi  = x_delta_sw2 - x_delta_sw2[0]
        ax2.plot(deltas_sw,  b_low, color=C_GEN, lw=2.2,
                 label=f"τ_A = {tau_A:.3f}")
        ax2.plot(deltas_sw2, b_hi,  color=C_AI,  lw=2.0, ls="--",
                 label=f"τ_A = {tau_A_hi:.3f}  (higher AI)")
        ax2.axhline(0, color="black", lw=0.8, alpha=0.4)
        ax2.axvline(delta, color=C_MIG, lw=1.3, ls=":", alpha=0.7)
        ax2.set_xlabel("Transferability  δ", fontsize=11)
        ax2.set_ylabel("ΣX_d(δ) − ΣX_d(0)  at t = T", fontsize=11)
        ax2.set_title("Marginal benefit of openness (relative to δ=0)", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

        slope_low = np.polyfit(deltas_sw,  b_low, 1)[0]
        slope_hi  = np.polyfit(deltas_sw2, b_hi,  1)[0]
        if slope_hi > slope_low:
            st.success(f"P2 supported: benefit slope steeper at higher τ_A "
                       f"({slope_hi:.3f} vs {slope_low:.3f}). "
                       "Openness matters more when AI is stronger.")
        else:
            st.warning(f"P2 not clearly supported (slopes: {slope_low:.3f} vs {slope_hi:.3f}). "
                       "Try τ_A above τ_A^c.")
    else:
        st.info("Check the box above to run the sweep. Results are cached after first run.")

# ─── Tab 5: Extension Guide ───────────────────────────────────────────────────
with tab5:
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

**The correct approach: spawn when it is individually rational to create a new domain.**

In credit equilibrium, each agent earns `G(X_d) / N_d` per period.  With N agents distributed
across all existing domains at the credit-equalized allocation, per-agent credit is:

>  c* = **ΣG(X_d) / N**

The pre-AI benchmark credit (no AI, one domain at steady state X_h0) is:

>  c₀ = **G(X_h0) / N**

**Endogenous spawn condition:**
>  **ΣG(X_d) < G(X_h0)**

Interpretation: agents create a new domain when the total credit available in existing domains
has fallen below the no-AI benchmark.  At that point, a new domain with inheritance
X_{{d',0}} = δ · ΣX_d offers positive credit that restores the pool above c₀ (if δ is high enough),
making spawning collectively beneficial.

**Why this fixes the cascade bug:**  at t = 0 with one domain at X_h0,
ΣG = G(X_h0) = G_h0 = {G_h0:.4f} — exactly at the threshold.  The condition `ΣG < G_h0`
is FALSE at initialisation, so no immediate spawn.  As AI erodes X_d, ΣG falls and eventually
crosses the threshold, spawning at the right time.  After each spawn, G(δ·ΣX_d) is added to
ΣG, which may push it back above G_h0 until further decay brings it down again.

**Current state:**  X_h0 = {X_h0:.3f},  G_h0 = {G_h0:.4f},
current ΣG = {sum(float(G(sim_o['domain_X'][i][~np.isnan(sim_o['domain_X'][i])][-1])) for i in range(n_dom) if (~np.isnan(sim_o['domain_X'][i])).any()):.4f},
domains spawned = {n_dom - 1}
""")

    with st.expander("🔀  Convergent inheritance — why ALL domains contribute"):
        st.markdown(f"""
**Chain inheritance (wrong):**  when d₁ spawns d₂, only d₁ contributes:
X_{{d₂,0}} = δ · X_{{d₁}}

This misses the fundamental point: when researchers from multiple fields jointly create a new
field, they bring knowledge from ALL their fields, not just the most recent one.

**Convergent inheritance (correct):**  ALL active domains contribute to the new one:
> **X_{{d',0}} = δ · ΣX_d = δ · (X_{{d₀}} + X_{{d₁}} + X_{{d₂}} + …)**

The parameter δ ∈ [0, 1] is the **knowledge transferability** — how much of the accumulated
domain knowledge survives the move to a new context.  δ = 0 → nothing transfers (each new domain
starts from scratch).  δ = 1 → perfect transfer (new domain starts with full collective knowledge).

**Key mathematical consequence:**  at each spawn, ΣX_d grows by factor **(1+δ)**:
- Before spawn:  ΣX_d = S
- New domain receives:  X_{{d',0}} = δ · S
- After spawn:  ΣX_d(new) = S + δ · S = **(1+δ) · S**

So after k spawns (starting from X_h0):  ΣX_d ≥ X_h0 · (1+δ)^k  (ignoring decay)

Current config: (1+δ) = {1+delta:.2f},  spawns so far = {n_dom - 1},
theoretical multiplier = {(1+delta)**(n_dom-1):.3f},  actual ΣX_d(T)/X_h0 = {sim_o['X_general'][-1]/X_h0:.3f}

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
{', '.join(f'd{i}: {sim_o["domain_N"][i][min(T-1, np.where(~np.isnan(sim_o["domain_X"][i]))[0][-1] if (~np.isnan(sim_o["domain_X"][i])).any() else 0)]:.1f}' for i in range(n_dom))} agents
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

γ = {gamma:.3f} currently.  {'No complementarity — purely additive recombination.' if gamma < 0.005 else f'Synergy bonus per spawn ≈ {gamma * sum(sim_o["domain_X"][i][np.where(~np.isnan(sim_o["domain_X"][i]))[0][-1]] * sim_o["domain_X"][j][np.where(~np.isnan(sim_o["domain_X"][j]))[0][-1]] for i in range(n_dom) for j in range(i+1, n_dom) if (~np.isnan(sim_o["domain_X"][i])).any() and (~np.isnan(sim_o["domain_X"][j])).any()):.3f}  (last spawn)' if n_dom > 1 else ''}
""")

    with st.expander("📐  Two predictions — mechanism to test"):
        st.markdown(f"""
**P1 — Open economy weakens or reverses AI-induced knowledge collapse**

*Mechanism:*  ΣX_d grows by factor (1+δ) at each spawn.  Even as individual domains collapse,
the cascade preserves and accumulates general knowledge.  With high δ and γ > 0,
ΣX_d can exceed X_h0 (full reversal of collapse, not just partial buffer).

*Why this is non-trivial:*  in the baseline closed economy, once X_d → 0 it stays there (absorbing).
The open economy breaks this: collapse in one domain triggers birth of another, and the new domain
inherits a fraction of the dying domain's knowledge.  The collapse is no longer absorbing.

*Current outcome:*  ΣX_d(T)/X_h0 = **{sim_o['X_general'][-1]/X_h0:.2f}**
{"→ Strong reversal: ΣX_d exceeds X_h0" if sim_o['X_general'][-1] > X_h0 else
 "→ Partial buffer: ΣX_d above collapse but below X_h0" if sim_o['X_general'][-1] > 0.3 * X_h0 else
 "→ Weak buffer: try higher δ or load strong-reversal config"}

*Test:*  Tab 3 (P1 — τ_A sweep) shows ΣX_d(T) open vs closed across AI capability levels.
P1 is supported when the green curve lies above the black curve past τ_A^c = {tau_c:.3f}.

---

**P2 — The benefit of openness is largest when AI capability is strongest**

*Mechanism:*  higher τ_A → faster domain erosion → trigger fires sooner → more spawns per unit
time → more (1+δ) multiplications of ΣX_d.  The open-economy advantage over closed therefore
*grows with τ_A* rather than being a fixed constant.

With γ > 0, a secondary amplification also operates: richer domains at spawn time produce larger
synergy bonuses (γ · Σ X_i · X_j grows with knowledge levels), making the benefit of openness
even more sensitive to AI capability.

*Why P2 matters for policy:*  it says openness (δ > 0) is a *complement* to AI capability,
not a substitute.  Societies with high cross-domain mobility benefit disproportionately more
in a world with strong AI.  The benefit of investing in knowledge portability (δ) is highest
exactly when AI pressure is greatest.

*Test:*  Tab 4 (P2 — δ sweep) shows ΣX_d(T) vs δ at two τ_A levels.  P2 is supported
when the slope of the curve is steeper at higher τ_A.

---
**Quick-reference: load the strong-reversal config (⚡ button)**

δ = 0.85, γ = 0.05, τ_A = 0.85, max_domains = 8

With endogenous trigger and these parameters:
- Each spawn multiplies ΣX_d by (1+0.85) = 1.85. After 3 spawns: 1.85³ ≈ 6.3×.
- τ_A = 0.85 > τ_A^c ({tau_c:.3f}): domains actively erode, triggering ΣG < G_h0 repeatedly.
- The endogenous trigger fires earlier (when knowledge is still relatively high) than an
  exogenous threshold would — giving richer inheritance at each spawn.
- γ = 0.05 adds pairwise synergy at each spawn, compounding the multiplier effect.
- max_domains = 8 allows the cascade to run 7 generations.
""")

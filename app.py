import base64
import json
import urllib.error
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Save the Fed",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ASSETS = Path("assets")

ACCOUNTABILITY = [
    # {"name": "Name here", "what": "What they did", "proof": "Link or citation text"},
]

# -------------------------
# Altair dark theme, forces transparent chart background and visible axes
# -------------------------
def _savefed_altair_theme():
    return {
        "config": {
            "background": "rgba(0,0,0,0)",
            "view": {"strokeOpacity": 0, "fill": "rgba(0,0,0,0)"},
            "axis": {
                "labelColor": "rgba(255,255,255,0.80)",
                "titleColor": "rgba(255,255,255,0.90)",
                "gridColor": "rgba(255,255,255,0.10)",
                "tickColor": "rgba(255,255,255,0.18)",
                "domainColor": "rgba(255,255,255,0.18)",
            },
            "legend": {
                "labelColor": "rgba(255,255,255,0.80)",
                "titleColor": "rgba(255,255,255,0.90)",
            },
            "title": {"color": "rgba(255,255,255,0.92)"},
        }
    }


alt.themes.register("savefed_dark", _savefed_altair_theme)
alt.themes.enable("savefed_dark")

# -------------------------
# Helpers
# -------------------------
def _mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def b64_image(path: Path) -> tuple[str | None, str]:
    if not path.exists():
        return None, "image/jpeg"
    return base64.b64encode(path.read_bytes()).decode("utf-8"), _mime(path)


def md(html: str):
    html = html.strip("\n")
    html = "\n".join(line.lstrip() for line in html.splitlines())
    st.markdown(html, unsafe_allow_html=True)


def compose_bg_css(url: str | None) -> str:
    parts = [
        "radial-gradient(1100px 700px at 15% 20%, rgba(122,166,255,0.20), transparent 60%)",
        "radial-gradient(900px 650px at 85% 20%, rgba(229,57,53,0.16), transparent 60%)",
        "linear-gradient(180deg, rgba(7,8,12,0.24), rgba(7,8,12,0.90))",
    ]
    if url:
        parts.append(f'url("{url}")')
    return ", ".join(parts)


# -------------------------
# Background images
# -------------------------
BG_FILES = {
    "hero": ASSETS / "hero.jpg",
    "explorer": ASSETS / "scenarios.jpg",
    "tracks": ASSETS / "scenarios.jpg",
    "cases": ASSETS / "case_studies.jpg",
    "why": ASSETS / "why.jpg",
    "trust": ASSETS / "trust.jpg",
    "about": ASSETS / "about.jpg",
}

BG_DATA: dict[str, str] = {}
for k, p in BG_FILES.items():
    b64, mime = b64_image(p)
    BG_DATA[k] = f"data:{mime};base64,{b64}" if b64 else ""

layer_a_default = compose_bg_css(BG_DATA.get("hero") or None)
layer_b_default = compose_bg_css(BG_DATA.get("explorer") or BG_DATA.get("hero") or None)

# -------------------------
# CSS
# -------------------------
CSS_TMPL = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
  --bg: #07080c;
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.76);
  --dim: rgba(255,255,255,0.62);
  --line: rgba(255,255,255,0.10);
}

html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}

header, footer { visibility: hidden; height: 0; }
#MainMenu { visibility: hidden; }
div[data-testid="stToolbar"] { visibility: hidden !important; height: 0 !important; }
div[data-testid="stDecoration"] { display: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }

.block-container {
  padding: 0 !important;
  max-width: none !important;
}

a, a:visited, a:hover, a:active {
  color: inherit !important;
  text-decoration: none !important;
}

.anchor {
  position: relative;
  top: -92px;
  height: 0;
}

/* Force scroll to stay enabled */
div[data-testid="stAppViewContainer"] {
  overflow-y: auto !important;
  color: var(--text) !important;
}

/* Make Streamlit text readable on dark backgrounds */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] p,
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] span,
div[data-testid="stText"] p,
div[data-testid="stText"] span {
  color: var(--text) !important;
}

div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] p {
  color: var(--muted) !important;
}

div[data-testid="stHelp"] * {
  color: var(--dim) !important;
}

/* Make Vega charts transparent, fixes white boxes and missing contrast */
div[data-testid="stVegaLiteChart"] {
  background: transparent !important;
}
div[data-testid="stVegaLiteChart"] > div {
  background: transparent !important;
}
div[data-testid="stVegaLiteChart"] svg {
  background: transparent !important;
}

/* ----- Cinematic fixed background stage ----- */
.bg-stage {
  position: fixed;
  inset: 0;
  z-index: 0;
  overflow: hidden;
  background: #07080c;
  pointer-events: none;
}

.bg-layer {
  position: absolute;
  inset: -10%;
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  transform: translate3d(0,0,0) scale(1.08);
  will-change: opacity, transform;
  filter: saturate(1.10) contrast(1.04) brightness(1.14);
  pointer-events: none;
}

.bg-layer.layer-a {
  opacity: 1;
  background-image: __LAYER_A__;
}

.bg-layer.layer-b {
  opacity: 0;
  background-image: __LAYER_B__;
}

@keyframes subtle_breathe {
  0%   { filter: saturate(1.08) contrast(1.03) brightness(1.10); }
  50%  { filter: saturate(1.12) contrast(1.05) brightness(1.18); }
  100% { filter: saturate(1.10) contrast(1.04) brightness(1.12); }
}

.bg-layer {
  animation: subtle_breathe 14s ease-in-out infinite;
}

.noise {
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: 0.06;
  z-index: 1;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)' opacity='.35'/%3E%3C/svg%3E");
}

.wrap {
  max-width: 1160px;
  margin: 0 auto;
  padding: 0 28px;
  position: relative;
  z-index: 3;
}

/* NAV */
.nav {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 999999;
  background: rgba(7,8,12,0.55);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--line);
}

.nav-inner {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 0;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  font-weight: 900;
  letter-spacing: -0.02em;
  color: rgba(255,255,255,0.94) !important;
}

.brand-icon {
  width: 38px; height: 38px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, rgba(229,57,53,0.90), rgba(122,166,255,0.85));
  box-shadow: 0 16px 50px rgba(0,0,0,0.45);
}

.nav-links {
  display: flex;
  gap: 18px;
  align-items: center;
}

.nav-links a {
  color: rgba(255,255,255,0.80) !important;
  font-size: 0.95rem;
  padding: 8px 10px;
  border-radius: 12px;
  transition: all 160ms ease;
}

.nav-links a:hover {
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.96) !important;
}

.nav-cta {
  padding: 9px 12px !important;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
}

.nav-cta:hover {
  background: rgba(255,255,255,0.10);
}

/* Sections */
.hero, .section-head, .section-body {
  position: relative;
  z-index: 2;
  background: transparent !important;
}

.hero::before, .section-head::before, .section-body::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 2;
  background:
    linear-gradient(180deg,
      rgba(7,8,12,0.70) 0%,
      rgba(7,8,12,0.38) 18%,
      rgba(7,8,12,0.34) 82%,
      rgba(7,8,12,0.74) 100%);
}

.hero {
  min-height: 92vh;
  display: flex;
  align-items: center;
}

.hero-inner {
  width: 100%;
  padding-top: 92px;
  padding-bottom: 56px;
}

.hero-grid {
  display: grid;
  grid-template-columns: 1.15fr 0.95fr;
  gap: 26px;
  align-items: center;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.86);
  font-size: 0.92rem;
  font-weight: 800;
}

.h1 {
  font-size: clamp(2.6rem, 4.1vw, 4.1rem);
  line-height: 1.05;
  letter-spacing: -0.045em;
  font-weight: 950;
  margin: 14px 0 12px 0;
  color: rgba(255,255,255,0.96) !important;
}

.subhead {
  color: rgba(255,255,255,0.84) !important;
  font-size: 1.10rem;
  line-height: 1.75;
  max-width: 68ch;
}

.cta-row {
  display: flex;
  gap: 12px;
  margin-top: 18px;
  flex-wrap: wrap;
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 12px 16px;
  border-radius: 14px;
  font-weight: 800;
  font-size: 0.98rem;
  border: 1px solid rgba(255,255,255,0.14);
  transition: all 180ms ease;
  color: rgba(255,255,255,0.92) !important;
  text-decoration: none !important;
}

.btn-ghost { background: rgba(255,255,255,0.06); }
.btn-ghost:hover { background: rgba(255,255,255,0.10); transform: translateY(-1px); }

.glass {
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.06);
  backdrop-filter: blur(18px);
  box-shadow: 0 22px 70px rgba(0,0,0,0.45);
  overflow: hidden;
}

.glass-head {
  padding: 18px;
  font-weight: 950;
  letter-spacing: -0.02em;
  color: rgba(255,255,255,0.94) !important;
}

.glass-body { padding: 0 18px 18px 18px; }

.track-card {
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.05);
  padding: 14px;
  margin-top: 12px;
}

.track-top { display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px; }
.track-name { font-weight: 950; letter-spacing: -0.02em; color: rgba(255,255,255,0.94); }
.track-badge {
  font-size: 0.86rem;
  color: rgba(255,255,255,0.78);
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(0,0,0,0.18);
}

.mini-text { color: rgba(255,255,255,0.82) !important; font-size: 0.95rem; line-height: 1.55; }

.section-head {
  padding: 84px 0 18px 0;
}

.section-body {
  padding: 18px 0 84px 0;
}

.h2 {
  font-size: 2.00rem;
  font-weight: 950;
  letter-spacing: -0.03em;
  margin: 0 0 10px 0;
  color: rgba(255,255,255,0.96) !important;
}

.lead {
  color: rgba(255,255,255,0.84) !important;
  font-size: 1.05rem;
  line-height: 1.80;
  max-width: 90ch;
}

.kicker {
  color: rgba(255,255,255,0.78) !important;
  font-size: 0.98rem;
  line-height: 1.70;
  max-width: 95ch;
}

.callout {
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.05);
  padding: 16px;
}

.small {
  color: rgba(255,255,255,0.76) !important;
  font-size: 0.95rem;
  line-height: 1.60;
}

ul.tight {
  margin: 8px 0 0 18px;
  padding: 0;
}
ul.tight li {
  margin: 6px 0;
  color: rgba(255,255,255,0.84) !important;
}

div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 12px 12px;
}
div[data-testid="stMetric"] * { color: rgba(255,255,255,0.90) !important; }

@media (max-width: 980px) {
  .hero-grid { grid-template-columns: 1fr; }
  .nav-links { display: none; }
}
</style>
"""

md(
    CSS_TMPL.replace("__LAYER_A__", layer_a_default).replace("__LAYER_B__", layer_b_default)
)

md(
    """
<div class="bg-stage">
  <div class="bg-layer layer-a"></div>
  <div class="bg-layer layer-b"></div>
</div>
<div class="noise"></div>
"""
)

# -------------------------
# NAV
# -------------------------
md(
    """
<div class="nav">
  <div class="wrap nav-inner">
    <div class="brand">
      <div class="brand-icon">🏛️</div>
      <div>Save the Fed</div>
    </div>
    <div class="nav-links">
      <a class="nav-cta" href="#explorer">Scenario explorer</a>
      <a href="#tracks">Tracks</a>
      <a href="#cases">Cases</a>
      <a href="#why">Why</a>
      <a href="#trust">Trust</a>
      <a href="#about">About</a>
    </div>
  </div>
</div>
"""
)

# -------------------------
# HERO
# -------------------------
md(
    """
<div class="hero" id="top">
  <div class="wrap hero-inner">
    <div class="hero-grid">

      <div style="position:relative; z-index:3;">
        <div class="pill">🏛️ The Fed is a guardrail of American prosperity</div>
        <div class="h1">If politicians control the Fed, your money gets weaker.</div>
        <div class="subhead">
          When leaders pressure the Fed to print, cut, and cover deficits, markets notice, prices move, and working people pay.
          This tool shows how that pressure has propagated in real data, and what tends to move next.
        </div>

        <div class="cta-row">
          <a class="btn btn-ghost" href="#explorer">Open the explorer</a>
          <a class="btn btn-ghost" href="#tracks">See the tracks</a>
          <a class="btn btn-ghost" href="#cases">See the cases</a>
        </div>

        <div style="height:14px;"></div>
        <div class="small">
          This helps you see risk before it hits your paycheck.
        </div>
      </div>

      <div class="glass" style="position:relative; z-index:3;">
        <div class="glass-head">Three tracks people can grasp fast</div>
        <div class="glass-body">

          <div class="track-card">
            <div class="track-top">
              <div class="track-name">Track A, independence holds</div>
              <div class="track-badge">Lower risk</div>
            </div>
            <div class="mini-text">Pressure stays contained, inflation expectations stay anchored, volatility stays calmer.</div>
          </div>

          <div class="track-card">
            <div class="track-top">
              <div class="track-name">Track B, rules get bent</div>
              <div class="track-badge">Drift</div>
            </div>
            <div class="mini-text">Markets price more uncertainty, the path gets harder to reverse.</div>
          </div>

          <div class="track-card">
            <div class="track-top">
              <div class="track-name">Track C, independence breaks</div>
              <div class="track-badge">High tail risk</div>
            </div>
            <div class="mini-text">Credibility takes a hit, inflation tail risk rises, volatility spikes under stress.</div>
          </div>

        </div>
      </div>

    </div>
  </div>
</div>
"""
)

# ============================================================
# Quant engine
# ============================================================

@st.cache_data(show_spinner=False)
def _fetch_fred_once(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [c.strip() for c in df.columns]
    if len(df.columns) < 2:
        raise ValueError("FRED returned an unexpected format.")
    date_col = df.columns[0]
    val_col = df.columns[1]
    out = df.rename(columns={date_col: "date", val_col: "value"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    return out


@st.cache_data(show_spinner=False)
def fetch_fred(series_id: str) -> pd.DataFrame:
    bases = [
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?cosd=1900-01-01&id={sid}",
    ]
    last_err: Exception | None = None
    for b in bases:
        url = b.format(sid=series_id)
        try:
            return _fetch_fred_once(url)
        except Exception as e:
            last_err = e
    if last_err is None:
        raise ValueError("FRED fetch failed.")
    raise last_err


def fetch_fred_fallback(series_ids: list[str]) -> tuple[pd.DataFrame, str]:
    last_err: Exception | None = None
    for sid in series_ids:
        try:
            return fetch_fred(sid), sid
        except urllib.error.HTTPError as e:
            last_err = e
        except Exception as e:
            last_err = e
    if last_err is None:
        raise ValueError("FRED fetch failed.")
    raise last_err


def to_monthly(df: pd.DataFrame, how: str = "mean") -> pd.Series:
    s = df.set_index("date")["value"].astype(float)
    if how == "last":
        return s.resample("MS").last()
    return s.resample("MS").mean()


@st.cache_data(show_spinner=False)
def build_monthly_panel() -> pd.DataFrame:
    epum = to_monthly(fetch_fred("EPUMONETARY"), how="mean").rename("EPU_MON")
    t5y5y = to_monthly(fetch_fred("T5YIFR"), how="mean").rename("INF5Y5Y")

    tp_raw, tp_used = fetch_fred_fallback(["THREEFYTP10", "ACMTP10"])
    tp10 = to_monthly(tp_raw, how="mean").rename("TP10")

    vix = to_monthly(fetch_fred("VIXCLS"), how="mean").rename("VIX")

    dgs10 = to_monthly(fetch_fred("DGS10"), how="mean")
    dgs2 = to_monthly(fetch_fred("DGS2"), how="mean")
    slope = (dgs10 - dgs2).rename("YC_SLOPE")

    df = pd.concat([epum, t5y5y, tp10, vix, slope], axis=1)
    df = df.dropna()
    df.attrs["tp_source"] = tp_used
    return df


def zscore(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = df.mean(axis=0).values
    sig = df.std(axis=0, ddof=0).replace(0, np.nan).values
    X = (df.values - mu) / sig
    return X, mu, sig


def fit_var_ols(X: np.ndarray, p: int) -> dict:
    T, k = X.shape
    if T <= p + 10:
        raise ValueError("Not enough observations.")

    Y = X[p:, :]
    Z_list = [np.ones((T - p, 1))]
    for i in range(1, p + 1):
        Z_list.append(X[p - i : T - i, :])
    Z = np.concatenate(Z_list, axis=1)

    B, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    Yhat = Z @ B
    E = Y - Yhat
    dof = max((T - p) - (1 + k * p), 1)
    Sigma = (E.T @ E) / dof

    c = B[0, :]
    A = []
    for i in range(p):
        Ai = B[1 + i * k : 1 + (i + 1) * k, :].T
        A.append(Ai)

    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        logdet = np.log(np.maximum(np.linalg.det(Sigma), 1e-12))

    ll = -0.5 * (T - p) * logdet
    n_params = k * (1 + k * p)
    aic = -2 * ll + 2 * n_params

    return {"p": p, "c": c, "A": A, "Sigma": Sigma, "aic": aic}


def select_var_lag(X: np.ndarray, p_max: int = 6) -> dict:
    best = None
    for p in range(1, p_max + 1):
        try:
            m = fit_var_ols(X, p)
        except Exception:
            continue
        if best is None or m["aic"] < best["aic"]:
            best = m
    if best is None:
        raise ValueError("VAR fitting failed.")
    return best


def deterministic_forecast(model: dict, x_hist: np.ndarray, shocks: np.ndarray) -> np.ndarray:
    p = model["p"]
    c = model["c"]
    A = model["A"]
    n_steps = shocks.shape[0]
    k = c.shape[0]

    out = np.zeros((n_steps, k), dtype=float)
    hist = x_hist.copy()
    for t in range(n_steps):
        x = c.copy()
        for i in range(1, p + 1):
            x += A[i - 1] @ hist[-i, :]
        x += shocks[t, :]
        out[t, :] = x
        hist = np.vstack([hist[1:, :], x.reshape(1, -1)])
    return out


def simulate_var(model: dict, x_hist: np.ndarray, shocks: np.ndarray, n_sims: int, seed: int) -> np.ndarray:
    p = model["p"]
    c = model["c"]
    A = model["A"]
    Sigma = model["Sigma"]
    n_steps = shocks.shape[0]
    k = c.shape[0]

    sims = np.zeros((n_sims, n_steps, k), dtype=float)
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(k))
    rng = np.random.default_rng(int(seed))

    for s in range(n_sims):
        hist = x_hist.copy()
        for t in range(n_steps):
            x = c.copy()
            for i in range(1, p + 1):
                x += A[i - 1] @ hist[-i, :]
            x += shocks[t, :]
            eps = L @ rng.standard_normal(k)
            x = x + eps
            sims[s, t, :] = x
            hist = np.vstack([hist[1:, :], x.reshape(1, -1)])

    return sims


def build_pressure_shock_path(horizon: int, intensity_sd: float, duration_m: int, persistence: float, insulation: float) -> np.ndarray:
    eff = intensity_sd * (1.0 - 0.85 * insulation)
    eff = max(eff, 0.0)

    s = np.zeros(horizon, dtype=float)
    dur = int(np.clip(duration_m, 1, horizon))
    s[:dur] = eff

    for t in range(dur, horizon):
        s[t] = s[t - 1] * float(np.clip(persistence, 0.0, 0.98))

    return s


def to_original_units(x_z: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return x_z * sig + mu


def credibility_score(path_df: pd.DataFrame) -> pd.Series:
    cols = ["INF5Y5Y", "TP10", "VIX", "YC_SLOPE"]

    roll_mean = path_df[cols].rolling(60, min_periods=12).mean()
    roll_std = path_df[cols].rolling(60, min_periods=12).std()
    z = (path_df[cols] - roll_mean) / roll_std
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    raw = 0.30 * z["INF5Y5Y"] + 0.25 * z["TP10"] + 0.30 * z["VIX"] + 0.15 * (-z["YC_SLOPE"])
    score = 50 + 18 * raw
    return score.clip(0, 100)


@st.cache_resource(show_spinner=False)
def get_engine():
    df = build_monthly_panel()
    X, mu, sig = zscore(df)
    model = select_var_lag(X, p_max=6)
    return df, X, mu, sig, model


def fan_chart(plot_df: pd.DataFrame, y_title: str):
    base = alt.Chart(plot_df).encode(x=alt.X("date:T", title=None))

    band = base.mark_area(opacity=0.25).encode(
        y=alt.Y("p10:Q", title=y_title),
        y2="p90:Q",
        tooltip=["date:T", "p10:Q", "p90:Q"],
    )

    l1 = base.mark_line(strokeWidth=2).encode(
        y="scenario:Q",
        tooltip=["date:T", "scenario:Q"],
    )

    l2 = base.mark_line(strokeDash=[4, 4], strokeWidth=2, opacity=0.85).encode(
        y="baseline:Q",
        tooltip=["date:T", "baseline:Q"],
    )

    chart = (band + l1 + l2).properties(height=250)
    st.altair_chart(chart, use_container_width=True)


# ============================================================
# Scenario Explorer
# ============================================================

md('<div class="anchor" id="explorer"></div>')
md(
    """
<div class="section-head">
  <div class="wrap">
    <div class="h2">Scenario Explorer</div>
    <div class="lead">
      Move the sliders and see how political pressure on the Fed has tended to spread through markets in real history.
      You get a baseline path, a stressed path, and uncertainty bands from simulation.
    </div>
    <div class="kicker">
      Public mode explains it in plain language.
      Expert mode shows the raw model details.
    </div>
  </div>
</div>
"""
)

padL, main, padR = st.columns([1, 6, 1])

with main:
    try:
        with st.spinner("Loading data and fitting the model"):
            df, X, mu, sig, model = get_engine()
    except Exception as e:
        st.error("Data load failed, this usually means a FRED series changed or the network blocked it.")
        st.write(str(e))
        st.stop()

    expert = st.toggle("Expert mode", value=False)
    seed = st.slider("Simulation seed", 1, 9999, 1337, 1)

    if expert:
        tp_src = df.attrs.get("tp_source", "unknown")
        st.caption(
            "Data window "
            + str(df.index.min().date())
            + " to "
            + str(df.index.max().date())
            + ", lag "
            + str(model["p"])
            + ", term premium source "
            + str(tp_src)
        )

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        intensity = st.slider(
            "Political pressure intensity",
            0.0,
            3.0,
            1.4,
            0.1,
            help="Higher means a bigger shock to monetary policy uncertainty.",
        )
    with colB:
        insulation = st.slider(
            "Institutional guardrails strength",
            0.0,
            1.0,
            0.55,
            0.05,
            help="Higher means stronger insulation, pressure translates less into uncertainty.",
        )
    with colC:
        duration = st.slider(
            "Pressure duration in months",
            1,
            24,
            6,
            1,
            help="How long the pressure stays intense.",
        )

    colD, colE, colF = st.columns([1, 1, 1])
    with colD:
        persistence = st.slider(
            "Pressure stickiness",
            0.0,
            0.98,
            0.65,
            0.01,
            help="After the intense period, how much pressure lingers.",
        )
    with colE:
        horizon = st.slider(
            "Forecast horizon in months",
            6,
            36,
            24,
            1,
        )
    with colF:
        sims_n = st.slider(
            "Uncertainty simulations",
            100,
            1200,
            500,
            50,
        )

    k = X.shape[1]
    shock0 = build_pressure_shock_path(horizon, intensity, duration, persistence, insulation)
    shocks = np.zeros((horizon, k), dtype=float)
    shocks[:, 0] = shock0

    p = model["p"]
    x_hist = X[-p:, :].copy()

    base_mean_z = deterministic_forecast(model, x_hist, np.zeros_like(shocks))
    scen_mean_z = deterministic_forecast(model, x_hist, shocks)

    sims = simulate_var(model, x_hist, shocks, n_sims=int(sims_n), seed=int(seed))
    q_lo = np.quantile(sims, 0.10, axis=0)
    q_hi = np.quantile(sims, 0.90, axis=0)

    base_mean = to_original_units(base_mean_z, mu, sig)
    scen_mean = to_original_units(scen_mean_z, mu, sig)
    lo = to_original_units(q_lo, mu, sig)
    hi = to_original_units(q_hi, mu, sig)

    future_idx = pd.date_range(df.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    cols = list(df.columns)

    base_df = pd.DataFrame(base_mean, index=future_idx, columns=cols)
    scen_df = pd.DataFrame(scen_mean, index=future_idx, columns=cols)
    lo_df = pd.DataFrame(lo, index=future_idx, columns=cols)
    hi_df = pd.DataFrame(hi, index=future_idx, columns=cols)

    score = credibility_score(pd.concat([df.tail(80), scen_df], axis=0)).loc[future_idx]
    base_score = credibility_score(pd.concat([df.tail(80), base_df], axis=0)).loc[future_idx]

    h12 = min(12, horizon) - 1

    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Inflation expectations",
        f"{scen_df['INF5Y5Y'].iloc[h12]:.2f}",
        f"{(scen_df['INF5Y5Y'].iloc[h12] - base_df['INF5Y5Y'].iloc[h12]):+.2f} after 12 months",
    )
    m2.metric(
        "Bond term premium",
        f"{scen_df['TP10'].iloc[h12]:.2f}",
        f"{(scen_df['TP10'].iloc[h12] - base_df['TP10'].iloc[h12]):+.2f} after 12 months",
    )
    m3.metric(
        "Market stress proxy",
        f"{scen_df['VIX'].iloc[h12]:.1f}",
        f"{(scen_df['VIX'].iloc[h12] - base_df['VIX'].iloc[h12]):+.1f} after 12 months",
    )
    m4.metric(
        "Credibility risk score",
        f"{score.iloc[h12]:.0f} out of 100",
        f"{(score.iloc[h12] - base_score.iloc[h12]):+.0f} after 12 months",
    )

    md(
        """
<div class="callout" style="margin-top:14px;">
  <div class="small">
    Plain meaning, higher pressure usually lifts inflation expectations and term premium and volatility.
    That combination is how a credibility loss shows up before the pain hits daily life.
  </div>
</div>
"""
    )

    st.markdown("")
    if expert:
        st.markdown("#### Outputs, scenario versus baseline, band is 10 to 90 percent")

        plot_inf = pd.DataFrame({"date": future_idx, "baseline": base_df["INF5Y5Y"].values, "scenario": scen_df["INF5Y5Y"].values, "p10": lo_df["INF5Y5Y"].values, "p90": hi_df["INF5Y5Y"].values})
        fan_chart(plot_inf, "Inflation expectations in percent")

        plot_tp = pd.DataFrame({"date": future_idx, "baseline": base_df["TP10"].values, "scenario": scen_df["TP10"].values, "p10": lo_df["TP10"].values, "p90": hi_df["TP10"].values})
        fan_chart(plot_tp, "Term premium in percent points")

        plot_vix = pd.DataFrame({"date": future_idx, "baseline": base_df["VIX"].values, "scenario": scen_df["VIX"].values, "p10": lo_df["VIX"].values, "p90": hi_df["VIX"].values})
        fan_chart(plot_vix, "VIX monthly average")

        plot_slope = pd.DataFrame({"date": future_idx, "baseline": base_df["YC_SLOPE"].values, "scenario": scen_df["YC_SLOPE"].values, "p10": lo_df["YC_SLOPE"].values, "p90": hi_df["YC_SLOPE"].values})
        fan_chart(plot_slope, "Yield curve slope, 10y minus 2y")
    else:
        st.markdown("#### The three signals most people should watch")

        plot_inf = pd.DataFrame({"date": future_idx, "baseline": base_df["INF5Y5Y"].values, "scenario": scen_df["INF5Y5Y"].values, "p10": lo_df["INF5Y5Y"].values, "p90": hi_df["INF5Y5Y"].values})
        fan_chart(plot_inf, "Inflation expectations in percent")

        plot_tp = pd.DataFrame({"date": future_idx, "baseline": base_df["TP10"].values, "scenario": scen_df["TP10"].values, "p10": lo_df["TP10"].values, "p90": hi_df["TP10"].values})
        fan_chart(plot_tp, "Borrowing cost pressure, term premium")

        plot_vix = pd.DataFrame({"date": future_idx, "baseline": base_df["VIX"].values, "scenario": scen_df["VIX"].values, "p10": lo_df["VIX"].values, "p90": hi_df["VIX"].values})
        fan_chart(plot_vix, "Market stress proxy, VIX")

    st.markdown("#### Credibility risk score over time")
    score_df = pd.DataFrame({"date": future_idx, "score": score.values, "baseline": base_score.values})
    sc = alt.Chart(score_df).encode(x=alt.X("date:T", title=None))
    sc_line = sc.mark_line(strokeWidth=3).encode(y=alt.Y("score:Q", title="Risk score from 0 to 100"))
    sc_base = sc.mark_line(strokeDash=[4, 4], strokeWidth=2, opacity=0.85).encode(y="baseline:Q")
    st.altair_chart((sc_line + sc_base).properties(height=220), use_container_width=True)

    export = pd.concat(
        [
            scen_df.add_prefix("scenario_"),
            base_df.add_prefix("baseline_"),
            lo_df.add_prefix("p10_"),
            hi_df.add_prefix("p90_"),
            score.rename("scenario_score"),
            base_score.rename("baseline_score"),
        ],
        axis=1,
    )
    st.download_button(
        "Download scenario output as CSV",
        data=export.to_csv(index=True).encode("utf-8"),
        file_name="save_the_fed_scenario_output.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# Tracks
# ============================================================

md('<div class="anchor" id="tracks"></div>')
md(
    """
<div class="section-head">
  <div class="wrap">
    <div class="h2">Tracks</div>
    <div class="lead">
      One click setups you can share.
      These are fixed parameter sets that drive the same model you just used.
    </div>
  </div>
</div>
"""
)

padL, mid2, padR = st.columns([1, 6, 1])
with mid2:
    presets = {
        "Track A, independence holds": dict(intensity=0.6, insulation=0.85, duration=4, persistence=0.35, horizon=24),
        "Track B, rules get bent": dict(intensity=1.4, insulation=0.55, duration=6, persistence=0.65, horizon=24),
        "Track C, independence breaks": dict(intensity=2.4, insulation=0.20, duration=10, persistence=0.85, horizon=24),
    }

    tabs = st.tabs(list(presets.keys()))
    for tab, (name, pr) in zip(tabs, presets.items()):
        with tab:
            df, X, mu, sig, model = get_engine()
            p = model["p"]
            x_hist = X[-p:, :].copy()
            k = X.shape[1]

            shock0 = build_pressure_shock_path(pr["horizon"], pr["intensity"], pr["duration"], pr["persistence"], pr["insulation"])
            shocks = np.zeros((pr["horizon"], k), dtype=float)
            shocks[:, 0] = shock0

            base = deterministic_forecast(model, x_hist, np.zeros_like(shocks))
            scen = deterministic_forecast(model, x_hist, shocks)

            base_u = to_original_units(base, mu, sig)
            scen_u = to_original_units(scen, mu, sig)

            idx = pd.date_range(df.index.max() + pd.offsets.MonthBegin(1), periods=pr["horizon"], freq="MS")
            cols = list(df.columns)
            base_df = pd.DataFrame(base_u, index=idx, columns=cols)
            scen_df = pd.DataFrame(scen_u, index=idx, columns=cols)

            h12 = 11
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Inflation expectations after 12 months", f"{scen_df['INF5Y5Y'].iloc[h12]:.2f}", f"{(scen_df['INF5Y5Y'].iloc[h12]-base_df['INF5Y5Y'].iloc[h12]):+.2f}")
            c2.metric("Term premium after 12 months", f"{scen_df['TP10'].iloc[h12]:.2f}", f"{(scen_df['TP10'].iloc[h12]-base_df['TP10'].iloc[h12]):+.2f}")
            c3.metric("VIX after 12 months", f"{scen_df['VIX'].iloc[h12]:.1f}", f"{(scen_df['VIX'].iloc[h12]-base_df['VIX'].iloc[h12]):+.1f}")

            scv = credibility_score(pd.concat([df.tail(80), scen_df], axis=0)).loc[idx]
            base_scv = credibility_score(pd.concat([df.tail(80), base_df], axis=0)).loc[idx]
            c4.metric("Risk score after 12 months", f"{scv.iloc[h12]:.0f} out of 100", f"{(scv.iloc[h12]-base_scv.iloc[h12]):+.0f}")

# ============================================================
# Cases, Why, Trust, About
# ============================================================

md('<div class="anchor" id="cases"></div>')
md(
    """
<div class="section-head">
  <div class="wrap">
    <div class="h2">Real world cases</div>
    <div class="lead">
      Countries that politicize their central bank get punished fast, and the punishment lands on ordinary people.
    </div>
  </div>
</div>
"""
)

padL, body, padR = st.columns([1, 6, 1])
with body:
    md(
        """
<div class="section-body">
  <div class="wrap">
    <div class="callout">
      <div class="small">
        The point is not that the United States becomes another country overnight.
        The point is that the first steps look the same, and markets respond early.
      </div>
      <ul class="tight">
        <li>Turkey, pressure on the central bank, inflation surge, currency slide, credibility trap.</li>
        <li>Argentina, repeated interference, expectations reset, inflation becomes structural.</li>
        <li>United Kingdom in the 1970s, financing pressure and credibility stress, policy space collapses.</li>
        <li>United States in the 1970s, loose policy and pressure, unanchored expectations, later pain to restore trust.</li>
      </ul>
    </div>
  </div>
</div>
"""
    )

md('<div class="anchor" id="why"></div>')
md(
    """
<div class="section-head">
  <div class="wrap">
    <div class="h2">Why this matters</div>
    <div class="lead">
      Central bank credibility shows up in rent, groceries, car payments, credit cards, and the ability to plan a life.
    </div>
  </div>
</div>
"""
)

padL, body, padR = st.columns([1, 6, 1])
with body:
    md(
        """
<div class="section-body">
  <div class="wrap">
    <div class="callout">
      <div class="small">
        The chain is simple.
      </div>
      <ul class="tight">
        <li>Pressure on the Fed raises uncertainty.</li>
        <li>Uncertainty raises premia, borrowing costs rise.</li>
        <li>If expectations drift, purchasing power falls.</li>
        <li>Restoring credibility often requires recession level pain.</li>
      </ul>
    </div>
  </div>
</div>
"""
    )

md('<div class="anchor" id="trust"></div>')
md(
    """
<div class="section-head">
  <div class="wrap">
    <div class="h2">Trust is a system, it can break</div>
    <div class="lead">
      Independence is a guardrail built to stop short term politics from hijacking the currency.
    </div>
  </div>
</div>
"""
)

padL, body, padR = st.columns([1, 6, 1])
with body:
    md(
        """
<div class="section-body">
  <div class="wrap">
    <div class="callout">
      <ul class="tight">
        <li>Savers get punished, cash loses value, the least able to hedge lose the most.</li>
        <li>Mortgages and business loans price higher risk.</li>
        <li>Politics gets angrier because people feel poorer.</li>
        <li>Every announcement gets discounted, policy becomes less effective.</li>
      </ul>
    </div>
  </div>
</div>
"""
    )

md('<div class="anchor" id="about"></div>')
md(
    """
<div class="section-head">
  <div class="wrap">
    <div class="h2">About</div>
    <div class="lead">
      This is an educational tool meant to persuade with real data.
      It is legible to normal people, and it stays inspectable for experts.
    </div>
  </div>
</div>
"""
)

padL, body, padR = st.columns([1, 6, 1])
with body:
    if ACCOUNTABILITY:
        st.markdown("#### Accountability list")
        for item in ACCOUNTABILITY:
            name = str(item.get("name", "")).strip()
            what = str(item.get("what", "")).strip()
            proof = str(item.get("proof", "")).strip()
            if not name and not what and not proof:
                continue
            st.markdown(f"**{name}**")
            if what:
                st.write(what)
            if proof:
                st.write(proof)
            st.markdown("---")
    else:
        md(
            """
<div class="section-body">
  <div class="wrap">
    <div class="callout">
      <div class="small">
        
      </div>
    </div>
  </div>
</div>
"""
        )

# -------------------------
# Background controller JS, event driven, avoids scroll lock and jank
# -------------------------
bg_json = json.dumps(BG_DATA)

JS_TMPL = r"""
<script>
(() => {
  const BG = __BG_JSON__;

  const ORDER = [
    { id: "top",      key: "hero" },
    { id: "explorer", key: "explorer" },
    { id: "tracks",   key: "tracks" },
    { id: "cases",    key: "cases" },
    { id: "why",      key: "why" },
    { id: "trust",    key: "trust" },
    { id: "about",    key: "about" },
  ];

  const win = window.parent;
  const doc = win.document;

  const layerA = doc.querySelector(".bg-layer.layer-a");
  const layerB = doc.querySelector(".bg-layer.layer-b");
  if (!layerA || !layerB) return;

  const scroller =
    doc.querySelector('div[data-testid="stAppViewContainer"]') ||
    doc.querySelector('section.main') ||
    doc.documentElement;

  const els = ORDER.map(o => doc.getElementById(o.id));

  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const ease = (t) => (t < 0.5) ? (4*t*t*t) : (1 - Math.pow(-2*t + 2, 3) / 2);

  const compose = (url) => {
    const parts = [
      "radial-gradient(1100px 700px at 15% 20%, rgba(122,166,255,0.20), transparent 60%)",
      "radial-gradient(900px 650px at 85% 20%, rgba(229,57,53,0.16), transparent 60%)",
      "linear-gradient(180deg, rgba(7,8,12,0.24), rgba(7,8,12,0.90))"
    ];
    if (url && url.length > 0) parts.push('url("' + url + '")');
    return parts.join(",");
  };

  const setLayer = (el, key) => {
    const url = BG[key] || "";
    el.style.backgroundImage = compose(url);
  };

  const THRESHOLD_FRAC = 0.12;
  const FADE_START = 0.78;
  const FADE_END = 0.99;
  const PARALLAX = 0.03;

  setLayer(layerA, ORDER[0].key);
  setLayer(layerB, ORDER[1].key);
  layerB.style.opacity = 0;

  let curKey = ORDER[0].key;
  let nextKey = ORDER[1].key;

  let ticking = false;
  let opacity = 0.0;

  const getScrollTop = () => {
    if (scroller === doc.documentElement) {
      return win.scrollY || doc.documentElement.scrollTop || 0;
    }
    return scroller.scrollTop || 0;
  };

  const getIndex = () => {
    const threshold = win.innerHeight * THRESHOLD_FRAC;
    let best = 0;
    for (let i = 0; i < ORDER.length; i++) {
      const el = els[i];
      if (!el) continue;
      const top = el.getBoundingClientRect().top;
      if (top <= threshold) best = i;
    }
    return best;
  };

  const update = () => {
    ticking = false;

    const i = getIndex();
    const cur = ORDER[i];
    const nxt = ORDER[Math.min(i + 1, ORDER.length - 1)];

    if (cur.key !== curKey) {
      curKey = cur.key;
      setLayer(layerA, curKey);
    }
    if (nxt.key !== nextKey) {
      nextKey = nxt.key;
      setLayer(layerB, nextKey);
    }

    const el = els[i];
    if (el) {
      const threshold = win.innerHeight * THRESHOLD_FRAC;
      const rect = el.getBoundingClientRect();
      const h = Math.max(el.offsetHeight || rect.height || 1, 1);

      let sectionProgress = clamp((threshold - rect.top) / h, 0, 1);
      let t = (sectionProgress - FADE_START) / Math.max(FADE_END - FADE_START, 1e-6);
      t = clamp(t, 0, 1);
      t = ease(t);

      opacity = t;
      layerB.style.opacity = opacity.toFixed(4);
    }

    const drift = -getScrollTop() * PARALLAX;
    const transform = "translate3d(0," + drift + "px,0) scale(1.08)";
    layerA.style.transform = transform;
    layerB.style.transform = transform;
  };

  const schedule = () => {
    if (ticking) return;
    ticking = true;
    win.requestAnimationFrame(update);
  };

  scroller.addEventListener("scroll", schedule, { passive: true });
  win.addEventListener("resize", schedule, { passive: true });

  schedule();
})();
</script>
"""

components.html(JS_TMPL.replace("__BG_JSON__", bg_json), height=0, width=0)

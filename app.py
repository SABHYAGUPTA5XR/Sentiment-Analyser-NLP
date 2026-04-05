import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import json

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infer import _load_model, predict


st.set_page_config(
    page_title="NeuralSense · Sentiment Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

/* ─── Reset & Base ─── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #020810 !important;
    color: #e2eaf7 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,210,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(0,255,160,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 40% 30% at 60% 40%, rgba(120,80,255,0.04) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ─── Scanline overlay ─── */
[data-testid="stMain"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,210,255,0.012) 2px,
        rgba(0,210,255,0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ─── Streamlit chrome removal ─── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="block-container"] { padding: 2.5rem 3rem !important; max-width: 1400px !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ─── Hero header ─── */
.ns-hero {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 2.8rem;
    padding-bottom: 1.4rem;
    border-bottom: 1px solid rgba(0,210,255,0.18);
    position: relative;
}
.ns-hero::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0;
    width: 120px; height: 1px;
    background: linear-gradient(90deg, #00d2ff, transparent);
    box-shadow: 0 0 12px #00d2ff88;
}
.ns-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -0.03em;
    color: #fff;
    line-height: 1;
}
.ns-logo span { color: #00d2ff; }
.ns-tag {
    font-size: 0.7rem;
    color: #00d2ff;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    opacity: 0.8;
    display: block;
    margin-top: 6px;
}
.ns-badge {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #00ffa3;
    border: 1px solid rgba(0,255,163,0.35);
    border-radius: 4px;
    padding: 4px 10px;
    background: rgba(0,255,163,0.06);
}

/* ─── Input panel ─── */
.ns-panel {
    background: rgba(6,16,36,0.85);
    border: 1px solid rgba(0,210,255,0.14);
    border-radius: 20px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
}
.ns-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d2ff88, transparent);
}
.ns-panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00d2ff;
    margin-bottom: 1rem;
}

/* ─── Streamlit textarea override ─── */
.stTextArea textarea {
    background: rgba(0,8,20,0.9) !important;
    border: 1px solid rgba(0,210,255,0.22) !important;
    border-radius: 12px !important;
    color: #e2eaf7 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.92rem !important;
    resize: none !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    caret-color: #00d2ff !important;
}
.stTextArea textarea:focus {
    border-color: rgba(0,210,255,0.55) !important;
    box-shadow: 0 0 0 3px rgba(0,210,255,0.08), 0 0 20px rgba(0,210,255,0.12) !important;
    outline: none !important;
}
.stTextArea label { display: none !important; }

/* ─── Example buttons ─── */
.stButton button {
    background: rgba(0,210,255,0.06) !important;
    border: 1px solid rgba(0,210,255,0.22) !important;
    border-radius: 10px !important;
    color: #a0c8e8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.18s ease !important;
}
.stButton button:hover {
    background: rgba(0,210,255,0.14) !important;
    border-color: rgba(0,210,255,0.5) !important;
    color: #00d2ff !important;
    box-shadow: 0 0 16px rgba(0,210,255,0.15) !important;
    transform: translateY(-1px) !important;
}

/* ─── Primary analyze button ─── */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #00d2ff18, #00ffa318) !important;
    border: 1px solid rgba(0,210,255,0.5) !important;
    color: #00d2ff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
    box-shadow: 0 0 24px rgba(0,210,255,0.12) !important;
    transition: all 0.2s ease !important;
}
.stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #00d2ff28, #00ffa328) !important;
    box-shadow: 0 0 40px rgba(0,210,255,0.25), 0 0 80px rgba(0,210,255,0.08) !important;
    transform: translateY(-2px) !important;
}

/* ─── Divider ─── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,210,255,0.12) !important;
    margin: 2.5rem 0 !important;
}

/* ─── Notes box ─── */
.ns-notes {
    background: rgba(0,8,20,0.7);
    border: 1px solid rgba(0,255,163,0.15);
    border-radius: 16px;
    padding: 1.6rem;
    height: 100%;
}
.ns-notes-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00ffa3;
    margin-bottom: 1rem;
}
.ns-note-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 0.75rem;
    font-size: 0.78rem;
    color: #7fa8c8;
    line-height: 1.5;
}
.ns-note-item::before {
    content: '⬡';
    color: #00ffa366;
    font-size: 0.65rem;
    margin-top: 2px;
    flex-shrink: 0;
}

/* ─── Spinner ─── */
.stSpinner > div {
    border-color: #00d2ff transparent transparent transparent !important;
}

/* ─── Metric / result cards ─── */
.ns-results-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00d2ff;
    margin-bottom: 1.4rem;
}

/* ─── Expander ─── */
.streamlit-expanderHeader {
    background: rgba(0,8,20,0.7) !important;
    border: 1px solid rgba(0,210,255,0.14) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #7fa8c8 !important;
}
.streamlit-expanderContent {
    background: rgba(0,4,12,0.9) !important;
    border: 1px solid rgba(0,210,255,0.1) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* Warning / info ─── */
.stAlert {
    background: rgba(0,210,255,0.06) !important;
    border: 1px solid rgba(0,210,255,0.25) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #a0c8e8 !important;
}

/* code block */
.stCode, .stCodeBlock {
    background: rgba(0,4,12,0.95) !important;
    border: 1px solid rgba(0,210,255,0.1) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ns-hero">
    <div>
        <div class="ns-logo">Neural<span>Sense</span></div>
        <span class="ns-tag">Multi-task Sentiment Intelligence Engine</span>
    </div>
    <div class="ns-badge">⬡ BERT-based · v1.0</div>
</div>
""", unsafe_allow_html=True)


# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    return _load_model()

def set_example(text):
    st.session_state["input_text"] = text

if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""


# ── Input row ────────────────────────────────────────────────────────────────
left, right = st.columns([1.4, 0.6], gap="large")

with left:
    st.markdown('<div class="ns-panel">', unsafe_allow_html=True)
    st.markdown('<div class="ns-panel-title">⬡ Input Signal</div>', unsafe_allow_html=True)
    st.text_area("Sentence", key="input_text", height=160,
                 placeholder="Paste or type a sentence to analyse its sentiment…")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.button("💬 Example 1", use_container_width=True, on_click=set_example,
                  args=("I absolutely love this product!!!",))
    with c2:
        st.button("💬 Example 2", use_container_width=True, on_click=set_example,
                  args=("This is so disappointing and frustrating.",))
    with c3:
        st.button("💬 Example 3", use_container_width=True, on_click=set_example,
                  args=("Could you please review this document?",))

    analyze = st.button("⬡ RUN ANALYSIS", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="ns-notes">', unsafe_allow_html=True)
    st.markdown('<div class="ns-notes-title">System Capabilities</div>', unsafe_allow_html=True)
    notes = [
        "Multi-task BERT-style sentiment model",
        "Special-token cue injection pipeline",
        "Per-class confidence aggregation",
        "Rule-based explanation cue extraction",
        "Shared CLI + UI inference pipeline",
    ]
    for n in notes:
        st.markdown(f'<div class="ns-note-item">{n}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Analysis output ──────────────────────────────────────────────────────────
if analyze:
    text = st.session_state["input_text"].strip()
    if not text:
        st.warning("⬡ No input detected — please enter a sentence.")
        st.stop()

    with st.spinner("Running inference…"):
        model, tokenizer, polarity_map, emotion_map, tone_map = load_assets()
        result = predict(text, model, tokenizer, polarity_map, emotion_map, tone_map)

    st.markdown("<hr>", unsafe_allow_html=True)
    confidence = float(result["confidence"])
    intensity_raw = result["intensity"]
    intensity_label = "N/A" if intensity_raw is None else str(intensity_raw)

    # Map labels to colours
    polarity_colors = {"positive": "#00ffa3", "negative": "#ff4d6d", "neutral": "#00d2ff"}
    emotion_colors  = {"joy": "#ffe066", "anger": "#ff4d6d", "sadness": "#7faaff",
                       "fear": "#c77dff", "surprise": "#00d2ff", "disgust": "#ff9f43",
                       "neutral": "#8899aa"}
    tone_colors     = {"formal": "#00d2ff", "informal": "#ffe066", "aggressive": "#ff4d6d",
                       "polite": "#00ffa3", "neutral": "#8899aa"}

    p_col = polarity_colors.get(str(result["polarity"]).lower(), "#00d2ff")
    e_col = emotion_colors.get(str(result["emotion"]).lower(), "#00d2ff")
    t_col = tone_colors.get(str(result["tone"]).lower(), "#00d2ff")

    # ── Radial gauge + metric dashboard (pure HTML/CSS/SVG/JS) ───────────────
    dash_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: transparent;
    font-family: 'JetBrains Mono', monospace;
    color: #e2eaf7;
  }}

  /* ── section label ── */
  .section-label {{
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #00d2ff;
    margin-bottom: 1rem;
  }}

  /* ── top row: 4 label cards + confidence gauge ── */
  .top-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr) 1.2fr;
    gap: 14px;
    margin-bottom: 18px;
  }}

  .metric-card {{
    background: rgba(6,16,36,0.9);
    border: 1px solid rgba(0,210,255,0.14);
    border-radius: 16px;
    padding: 18px 16px;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 110px;
    transition: border-color 0.3s, box-shadow 0.3s;
    animation: fadeUp 0.5s ease both;
  }}
  .metric-card:hover {{
    border-color: rgba(0,210,255,0.35);
    box-shadow: 0 4px 24px rgba(0,210,255,0.10);
  }}
  .metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent);
    box-shadow: 0 0 10px var(--accent);
    opacity: 0.7;
  }}
  .metric-card:nth-child(1) {{ --accent: {p_col}; animation-delay: 0.05s; }}
  .metric-card:nth-child(2) {{ --accent: {e_col}; animation-delay: 0.12s; }}
  .metric-card:nth-child(3) {{ --accent: {t_col}; animation-delay: 0.18s; }}
  .metric-card:nth-child(4) {{ --accent: #c77dff; animation-delay: 0.24s; }}

  .card-label {{
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(160,200,230,0.6);
    margin-bottom: 10px;
  }}
  .card-value {{
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    color: var(--accent);
    text-transform: capitalize;
    text-shadow: 0 0 20px var(--accent);
    line-height: 1;
  }}
  .card-sub {{
    font-size: 0.6rem;
    color: rgba(160,200,230,0.4);
    margin-top: 8px;
    letter-spacing: 0.05em;
  }}

  /* ── Confidence gauge ── */
  .gauge-card {{
    background: rgba(6,16,36,0.9);
    border: 1px solid rgba(0,210,255,0.14);
    border-radius: 16px;
    padding: 18px 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 110px;
    animation: fadeUp 0.5s 0.3s ease both;
    position: relative;
    overflow: hidden;
  }}
  .gauge-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00d2ff, #00ffa3);
    box-shadow: 0 0 10px #00d2ff88;
  }}
  .gauge-label {{
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(160,200,230,0.6);
    margin-bottom: 10px;
    align-self: flex-start;
  }}
  .gauge-svg {{ width: 120px; height: 66px; }}
  .gauge-number {{
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.25rem;
    color: #00d2ff;
    text-shadow: 0 0 20px #00d2ff88;
    margin-top: 4px;
  }}

  /* ── Bottom row: bar charts for categorical dist ── */
  .bottom-row {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
    margin-bottom: 18px;
  }}
  .bar-card {{
    background: rgba(6,16,36,0.9);
    border: 1px solid rgba(0,210,255,0.1);
    border-radius: 16px;
    padding: 18px;
    animation: fadeUp 0.5s ease both;
  }}
  .bar-card:nth-child(1) {{ animation-delay: 0.35s; }}
  .bar-card:nth-child(2) {{ animation-delay: 0.42s; }}
  .bar-card:nth-child(3) {{ animation-delay: 0.49s; }}
  .bar-card-title {{
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(160,200,230,0.6);
    margin-bottom: 12px;
  }}
  .bar-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 9px;
  }}
  .bar-name {{
    font-size: 0.68rem;
    color: rgba(200,220,240,0.7);
    width: 72px;
    flex-shrink: 0;
    text-transform: capitalize;
  }}
  .bar-track {{
    flex: 1;
    height: 6px;
    background: rgba(0,210,255,0.08);
    border-radius: 999px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 999px;
    transform-origin: left;
    transform: scaleX(0);
    transition: transform 0.9s cubic-bezier(0.22,1,0.36,1);
    box-shadow: 0 0 6px var(--bar-color);
  }}
  .bar-pct {{
    font-size: 0.62rem;
    color: rgba(160,200,230,0.5);
    width: 32px;
    text-align: right;
    flex-shrink: 0;
  }}

  /* ── Token chips ── */
  .chips-card {{
    background: rgba(6,16,36,0.9);
    border: 1px solid rgba(0,210,255,0.1);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 14px;
    animation: fadeUp 0.5s 0.55s ease both;
  }}
  .chips-title {{
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(160,200,230,0.6);
    margin-bottom: 12px;
  }}
  .chip {{
    display: inline-block;
    margin: 4px 5px 4px 0;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    letter-spacing: 0.04em;
    border: 1px solid;
    transition: transform 0.15s, box-shadow 0.15s;
    cursor: default;
    animation: popIn 0.4s ease both;
  }}
  .chip:hover {{
    transform: translateY(-2px) scale(1.04);
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
  }}
  .chip-cue {{
    background: rgba(0,210,255,0.08);
    border-color: rgba(0,210,255,0.35);
    color: #00d2ff;
    box-shadow: 0 0 8px rgba(0,210,255,0.12);
  }}
  .chip-tok {{
    background: rgba(0,255,163,0.06);
    border-color: rgba(0,255,163,0.3);
    color: #00ffa3;
    box-shadow: 0 0 8px rgba(0,255,163,0.10);
  }}
  .chip-empty {{
    color: rgba(160,200,230,0.35);
    font-size: 0.72rem;
    font-style: italic;
  }}

  /* ── Animations ── */
  @keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(18px); }}
    to   {{ opacity:1; transform:translateY(0); }}
  }}
  @keyframes popIn {{
    0%   {{ opacity:0; transform:scale(0.6); }}
    70%  {{ transform:scale(1.08); }}
    100% {{ opacity:1; transform:scale(1); }}
  }}
</style>
</head>
<body>

<div class="section-label">⬡ Inference Results</div>

<!-- Top row: 4 metric cards + gauge -->
<div class="top-row">
  <div class="metric-card">
    <div class="card-label">Polarity</div>
    <div class="card-value">{result['polarity']}</div>
    <div class="card-sub">Sentiment direction</div>
  </div>
  <div class="metric-card">
    <div class="card-label">Emotion</div>
    <div class="card-value">{result['emotion']}</div>
    <div class="card-sub">Detected affect</div>
  </div>
  <div class="metric-card">
    <div class="card-label">Tone</div>
    <div class="card-value">{result['tone']}</div>
    <div class="card-sub">Communicative style</div>
  </div>
  <div class="metric-card">
    <div class="card-label">Intensity</div>
    <div class="card-value">{intensity_label}</div>
    <div class="card-sub">Signal strength</div>
  </div>

  <!-- Confidence donut gauge -->
  <div class="gauge-card">
    <div class="gauge-label">Confidence</div>
    <svg class="gauge-svg" viewBox="0 0 120 66" fill="none" xmlns="http://www.w3.org/2000/svg">
      <!-- track arc -->
      <path d="M10,60 A50,50 0 0,1 110,60" stroke="rgba(0,210,255,0.1)" stroke-width="8" stroke-linecap="round" fill="none"/>
      <!-- fill arc (animated via JS) -->
      <path id="gauge-arc" d="M10,60 A50,50 0 0,1 110,60"
            stroke="url(#gGrad)" stroke-width="8" stroke-linecap="round" fill="none"
            stroke-dasharray="157" stroke-dashoffset="157"/>
      <defs>
        <linearGradient id="gGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#00d2ff"/>
          <stop offset="100%" stop-color="#00ffa3"/>
        </linearGradient>
      </defs>
      <!-- glow -->
      <path d="M10,60 A50,50 0 0,1 110,60" stroke="url(#gGrad)"
            stroke-width="8" stroke-linecap="round" fill="none"
            stroke-dasharray="157" stroke-dashoffset="157"
            opacity="0.35" filter="url(#blur)" id="gauge-glow"/>
      <filter id="blur"><feGaussianBlur stdDeviation="3"/></filter>
    </svg>
    <div class="gauge-number" id="gauge-num">0.000</div>
  </div>
</div>

<!-- Bottom row: bar charts -->
<div class="bottom-row">
  <!-- Polarity bars (heuristic visual distribution) -->
  <div class="bar-card">
    <div class="bar-card-title">Polarity spectrum</div>
    <div id="polarity-bars"></div>
  </div>
  <!-- Explanation cues -->
  <div class="bar-card">
    <div class="bar-card-title">⬡ Explanation cues</div>
    <div id="cue-chips"></div>
  </div>
  <!-- Special tokens -->
  <div class="bar-card">
    <div class="bar-card-title">⬡ Special tokens</div>
    <div id="token-chips"></div>
  </div>
</div>

<script>
const CONF = {confidence};
const POLARITY = "{result['polarity']}".toLowerCase();
const EMOTION  = "{result['emotion']}".toLowerCase();
const TONE     = "{result['tone']}".toLowerCase();

// Cues & tokens from Python
const CUES   = {json.dumps(result.get('explanation') or [])};
const TOKENS = {json.dumps(result.get('special_tokens') or [])};

// ─── Gauge animation ───────────────────────────────────────────────────────
const totalArc = 157; // half-circle circumference for r=50 semicircle
let start = null;
const duration = 1100;
function animateGauge(ts) {{
  if (!start) start = ts;
  const t = Math.min((ts - start) / duration, 1);
  const ease = 1 - Math.pow(1 - t, 4);
  const offset = totalArc * (1 - CONF * ease);
  document.getElementById('gauge-arc').setAttribute('stroke-dashoffset', offset);
  document.getElementById('gauge-glow').setAttribute('stroke-dashoffset', offset);
  document.getElementById('gauge-num').textContent = (CONF * ease).toFixed(3);
  if (t < 1) requestAnimationFrame(animateGauge);
}}
requestAnimationFrame(animateGauge);

// ─── Polarity spectrum bars ────────────────────────────────────────────────
const polarityData = {{
  'positive': {{ color: '#00ffa3', weight: POLARITY === 'positive' ? CONF : (1 - CONF) * 0.28 }},
  'neutral':  {{ color: '#00d2ff', weight: POLARITY === 'neutral'  ? CONF : (1 - CONF) * 0.38 }},
  'negative': {{ color: '#ff4d6d', weight: POLARITY === 'negative' ? CONF : (1 - CONF) * 0.22 }},
}};
// Normalise
const pSum = Object.values(polarityData).reduce((a, b) => a + b.weight, 0);
const pEl = document.getElementById('polarity-bars');
Object.entries(polarityData).forEach(([name, d], i) => {{
  const pct = d.weight / pSum;
  pEl.innerHTML += `
    <div class="bar-row">
      <div class="bar-name">${{name}}</div>
      <div class="bar-track">
        <div class="bar-fill" style="--bar-color:${{d.color}};background:${{d.color}};width:${{(pct*100).toFixed(1)}}%;"
             data-pct="${{pct}}"></div>
      </div>
      <div class="bar-pct">${{(pct*100).toFixed(0)}}%</div>
    </div>`;
}});

// ─── Cue chips ────────────────────────────────────────────────────────────
const cueEl = document.getElementById('cue-chips');
if (CUES.length === 0) {{
  cueEl.innerHTML = '<span class="chip-empty">No rule-based cues detected.</span>';
}} else {{
  CUES.forEach((c, i) => {{
    cueEl.innerHTML += `<span class="chip chip-cue" style="animation-delay:${{i*0.07}}s">${{c}}</span>`;
  }});
}}

// ─── Token chips ──────────────────────────────────────────────────────────
const tokEl = document.getElementById('token-chips');
if (TOKENS.length === 0) {{
  tokEl.innerHTML = '<span class="chip-empty">No special tokens injected.</span>';
}} else {{
  TOKENS.forEach((t, i) => {{
    tokEl.innerHTML += `<span class="chip chip-tok" style="animation-delay:${{i*0.07+0.1}}s">${{t}}</span>`;
  }});
}}

// ─── Animate bars ─────────────────────────────────────────────────────────
setTimeout(() => {{
  document.querySelectorAll('.bar-fill').forEach(el => {{
    el.style.transform = 'scaleX(1)';
  }});
}}, 200);
</script>
</body>
</html>
"""

    components.html(dash_html, height=480, scrolling=False)

    # ── Expanders ─────────────────────────────────────────────────────────────
    with st.expander("⬡ View generated model text"):
        st.code(result["model_text"], language="text")

    with st.expander("⬡ View raw result dictionary"):
        st.json(result)
import os
import io
from datetime import datetime

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexTax PK — Pakistan Tax Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

<style>
:root{
  --bg-0:#070b14;
  --bg-1:#0b1220;
  --bg-2:#111a2e;
  --panel:#0f172a;
  --panel-2:#131d33;
  --border:rgba(212,175,55,0.12);
  --border-strong:rgba(212,175,55,0.28);
  --ink:#e8edf5;
  --ink-soft:#9aa7bd;
  --ink-mute:#6b7794;
  --gold:#d4af37;
  --gold-soft:#e8c870;
  --emerald:#10b981;
  --emerald-soft:#34d399;
  --danger:#f87171;
  --shadow: 0 30px 80px -20px rgba(0,0,0,0.6), 0 8px 24px -8px rgba(0,0,0,0.4);
}

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(212,175,55,0.08), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(16,185,129,0.06), transparent 55%),
    linear-gradient(180deg, #070b14 0%, #0a1020 100%);
  color: var(--ink);
}
.block-container{ padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1200px; }

/* Hide Streamlit chrome safely */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* Fix header z-index — sidebar toggle accessible */
header[data-testid="stHeader"] {
  z-index: 0 !important;
  background: transparent !important;
}

/* ── Tab Styling ─────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.02) !important;
  border-radius: 16px !important;
  padding: 6px !important;
  border: 1px solid var(--border) !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 12px !important;
  color: var(--ink-mute) !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  letter-spacing: .5px !important;
  padding: 10px 20px !important;
  border: none !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(180deg, rgba(212,175,55,0.18), rgba(212,175,55,0.08)) !important;
  color: var(--gold-soft) !important;
  border: 1px solid var(--border-strong) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Top Brand Bar ─────────────────────────────────────── */
.brandbar{
  display:flex; align-items:center; justify-content:space-between;
  padding: 14px 22px; border-radius: 16px;
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid var(--border);
  backdrop-filter: blur(14px);
  margin-bottom: 22px;
}
.brand{ display:flex; align-items:center; gap:14px; }
.brand-mark{
  width:46px; height:46px; border-radius:12px;
  background: conic-gradient(from 220deg, #d4af37, #b8860b, #d4af37, #f6e27a, #d4af37);
  display:grid; place-items:center;
  color:#0b1220; font-size:22px;
  box-shadow: 0 10px 30px rgba(212,175,55,0.25), inset 0 0 0 1px rgba(0,0,0,0.2);
}
.brand-title{ font-family:'Cormorant Garamond', serif; font-weight:700; font-size:22px; letter-spacing:.3px; color:var(--ink); line-height:1; }
.brand-title small{ display:block; font-family:'Plus Jakarta Sans',sans-serif; font-weight:500; font-size:11px; letter-spacing:3px; color:var(--gold-soft); text-transform:uppercase; margin-top:6px;}
.brand-pills{ display:flex; gap:8px; }
.pill{
  font-size:11px; font-weight:600; padding:6px 12px; border-radius:999px;
  border:1px solid var(--border-strong);
  color:var(--gold-soft); letter-spacing:.5px;
  background: rgba(212,175,55,0.06);
}
.pill.live::before{
  content:""; display:inline-block; width:7px; height:7px; border-radius:50%;
  background: var(--emerald-soft); margin-right:7px; vertical-align:middle;
  box-shadow: 0 0 12px var(--emerald-soft);
  animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse{ 0%,100%{opacity:1} 50%{opacity:.4} }

/* ── Hero ──────────────────────────────────────────────── */
.hero{
  position:relative; overflow:hidden;
  border-radius: 22px; padding: 38px 42px;
  background:
    radial-gradient(600px 240px at 90% -10%, rgba(212,175,55,0.18), transparent 60%),
    radial-gradient(500px 220px at 10% 110%, rgba(16,185,129,0.12), transparent 60%),
    linear-gradient(180deg, #0d1426, #0b1220);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  margin-bottom: 24px;
}
.hero::before{
  content:""; position:absolute; inset:0;
  background-image:
    linear-gradient(rgba(212,175,55,0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(212,175,55,0.06) 1px, transparent 1px);
  background-size: 28px 28px;
  mask-image: radial-gradient(600px 300px at 80% 20%, #000 30%, transparent 70%);
  pointer-events:none;
}
.hero-eyebrow{
  display:inline-flex; align-items:center; gap:8px;
  font-size:11px; font-weight:700; letter-spacing:3px; text-transform:uppercase;
  color: var(--gold-soft);
  padding:6px 12px; border-radius:999px;
  background: rgba(212,175,55,0.08); border:1px solid var(--border-strong);
}
.hero h1{
  font-family:'Cormorant Garamond', serif;
  font-size: 52px; line-height:1.05; font-weight:700;
  margin: 18px 0 10px; color: var(--ink);
  letter-spacing: -0.5px;
}
.hero h1 .accent{
  background: linear-gradient(90deg, #f6e27a 0%, #d4af37 50%, #b8860b 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
  font-style: italic;
}
.hero p{ color: var(--ink-soft); font-size:16px; max-width: 680px; line-height:1.7; margin:0; }
.hero-meta{ display:flex; gap:24px; margin-top:24px; flex-wrap:wrap; }
.meta-chip{ display:flex; align-items:center; gap:10px; color:var(--ink-soft); font-size:13px; }
.meta-chip i{ color: var(--gold); }

/* ── Section Headings ──────────────────────────────────── */
.section-label{
  display:flex; align-items:center; gap:12px;
  font-size:11px; font-weight:700; letter-spacing:3px; text-transform:uppercase;
  color: var(--gold-soft); margin: 8px 0 14px;
}
.section-label::after{
  content:""; flex:1; height:1px;
  background: linear-gradient(90deg, var(--border-strong), transparent);
}

/* ── Suggest Cards ─────────────────────────────────────── */
.suggest-grid .stButton > button{
  width:100%; text-align:left !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.01)) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  color: var(--ink) !important;
  padding: 18px 18px !important;
  font-weight: 500 !important; font-size: 14px !important;
  white-space: normal !important; line-height:1.5 !important;
  transition: all .25s ease !important;
  box-shadow: none !important;
}
.suggest-grid .stButton > button:hover{
  border-color: var(--border-strong) !important;
  background: linear-gradient(180deg, rgba(212,175,55,0.08), rgba(212,175,55,0.02)) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 14px 30px -14px rgba(212,175,55,0.35) !important;
}

/* ── Ask Form ──────────────────────────────────────────── */
.ask-shell{
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border:1px solid var(--border);
  border-radius: 20px;
  padding: 22px;
  box-shadow: var(--shadow);
  margin: 6px 0 22px;
}
.stTextArea textarea, .stTextInput input{
  background: #0a1222 !important;
  border:1px solid var(--border) !important;
  border-radius: 14px !important;
  color: var(--ink) !important;
  font-size: 15px !important;
  padding: 16px 18px !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  min-height: 64px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus{
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px rgba(212,175,55,0.15) !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder{ color: var(--ink-mute) !important; }

/* Number input */
.stNumberInput input {
  background: #0a1222 !important;
  border:1px solid var(--border) !important;
  border-radius: 14px !important;
  color: var(--ink) !important;
  font-size: 15px !important;
  padding: 12px 18px !important;
}
.stNumberInput input:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px rgba(212,175,55,0.15) !important;
}

/* Select box */
.stSelectbox > div > div {
  background: #0a1222 !important;
  border:1px solid var(--border) !important;
  border-radius: 14px !important;
  color: var(--ink) !important;
}

/* Radio */
.stRadio > div { gap: 10px !important; }
.stRadio label { color: var(--ink-soft) !important; font-size: 14px !important; }

/* Primary CTA button */
div[data-testid="stForm"] .stButton > button,
.ask-cta .stButton > button,
.calc-cta .stButton > button {
  background: linear-gradient(180deg, #e8c870, #d4af37 55%, #b8860b) !important;
  color:#0b1220 !important;
  border: 1px solid #f6e27a !important;
  border-radius: 14px !important;
  padding: 14px 24px !important;
  font-weight: 800 !important; font-size: 14px !important;
  letter-spacing: .6px !important; text-transform: uppercase !important;
  width: 100% !important;
  box-shadow: 0 12px 28px -10px rgba(212,175,55,0.55), inset 0 1px 0 rgba(255,255,255,0.35) !important;
  transition: all .2s ease !important;
}
div[data-testid="stForm"] .stButton > button:hover,
.calc-cta .stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 18px 36px -10px rgba(212,175,55,0.75), inset 0 1px 0 rgba(255,255,255,0.4) !important;
}

/* ── PDF Download Button ───────────────────────────────── */
.pdf-btn .stDownloadButton > button {
  background: linear-gradient(180deg, rgba(212,175,55,0.15), rgba(212,175,55,0.06)) !important;
  color: var(--gold-soft) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 12px !important;
  padding: 10px 20px !important;
  font-weight: 700 !important; font-size: 13px !important;
  letter-spacing: .5px !important;
  transition: all .2s ease !important;
  width: 100% !important;
}
.pdf-btn .stDownloadButton > button:hover {
  background: linear-gradient(180deg, rgba(212,175,55,0.25), rgba(212,175,55,0.12)) !important;
  border-color: var(--gold) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 20px -8px rgba(212,175,55,0.4) !important;
}

/* ── Answer Card ───────────────────────────────────────── */
.answer-card{
  position:relative;
  background:
    radial-gradient(500px 200px at 0% 0%, rgba(212,175,55,0.08), transparent 60%),
    linear-gradient(180deg, #0d1528, #0a1120);
  border: 1px solid var(--border-strong);
  border-radius: 22px;
  padding: 28px 30px;
  margin-top: 18px;
  color: var(--ink);
  font-size: 15.5px; line-height: 1.8;
  box-shadow: var(--shadow);
  overflow:hidden;
}
.answer-card::before{
  content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
  background: linear-gradient(180deg, #f6e27a, #d4af37, #b8860b);
}
.answer-head{
  display:flex; align-items:center; justify-content:space-between;
  margin-bottom: 18px; padding-bottom: 14px;
  border-bottom: 1px dashed var(--border);
}
.answer-head .left{ display:flex; align-items:center; gap:12px; }
.seal{
  width:38px; height:38px; border-radius:50%;
  display:grid; place-items:center; color:#0b1220;
  background: radial-gradient(circle at 30% 30%, #f6e27a, #d4af37 60%, #8a6a10);
  border: 2px solid rgba(255,255,255,0.15);
  box-shadow: 0 6px 16px rgba(212,175,55,0.3);
  font-size: 16px;
}
.answer-title{ font-family:'Cormorant Garamond',serif; font-size:22px; font-weight:700; color:var(--gold-soft); line-height:1;}
.answer-sub{ font-size:11px; letter-spacing:2px; color:var(--ink-mute); text-transform:uppercase; margin-top:6px;}
.answer-tag{
  font-size:11px; font-weight:700; padding:6px 12px; border-radius:999px;
  background: rgba(16,185,129,0.1); color: var(--emerald-soft);
  border: 1px solid rgba(16,185,129,0.3);
}
.answer-body{ color: var(--ink); }
.answer-body p, .answer-body li{ color: var(--ink); }
.answer-body strong{ color: var(--gold-soft); }

/* ── Tax Calculator Card ───────────────────────────────── */
.calc-card{
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 28px 30px;
  margin-bottom: 20px;
}
.calc-result{
  position:relative; overflow:hidden;
  background:
    radial-gradient(500px 200px at 0% 0%, rgba(212,175,55,0.1), transparent 60%),
    linear-gradient(180deg, #0d1528, #0a1120);
  border: 1px solid var(--border-strong);
  border-radius: 22px;
  padding: 28px 30px;
  margin-top: 20px;
  box-shadow: var(--shadow);
}
.calc-result::before{
  content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
  background: linear-gradient(180deg, #f6e27a, #d4af37, #b8860b);
}
.result-grid{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin: 20px 0;
}
.result-box{
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
  text-align: center;
}
.result-box .rb-label{
  font-size: 10px; font-weight: 700; letter-spacing: 2px;
  text-transform: uppercase; color: var(--ink-mute); margin-bottom: 8px;
}
.result-box .rb-value{
  font-family: 'Cormorant Garamond', serif;
  font-size: 22px; font-weight: 700; color: var(--gold-soft);
  line-height: 1;
}
.result-box .rb-value.green{ color: var(--emerald-soft); }
.result-box .rb-value.red{ color: var(--danger); }
.slab-table{
  width: 100%; border-collapse: collapse; margin-top: 16px;
  font-size: 13px;
}
.slab-table th{
  background: rgba(212,175,55,0.08);
  color: var(--gold-soft); font-weight: 700;
  padding: 10px 14px; text-align: left;
  border-bottom: 1px solid var(--border-strong);
  font-size: 11px; letter-spacing: 1px; text-transform: uppercase;
}
.slab-table td{
  padding: 10px 14px; color: var(--ink-soft);
  border-bottom: 1px solid var(--border);
}
.slab-table tr.active-slab td{
  color: var(--gold-soft);
  background: rgba(212,175,55,0.06);
  font-weight: 600;
}
.slab-table tr:last-child td{ border-bottom: none; }

/* ── Source Documents ──────────────────────────────────── */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--ink-soft) !important;
  font-weight: 600 !important;
}
.context-box{
  background: #0a1222;
  border: 1px solid var(--border);
  border-left: 3px solid var(--gold);
  border-radius: 12px;
  padding: 16px 18px;
  font-size: 13.5px; line-height:1.7;
  color: var(--ink-soft);
  margin: 12px 0;
  font-family: 'JetBrains Mono', monospace;
}
.context-box .src-label{
  display:inline-block; font-family:'Plus Jakarta Sans',sans-serif;
  font-size:10px; font-weight:700; letter-spacing:2px; color: var(--gold-soft);
  text-transform:uppercase; margin-bottom:8px;
}

/* ── Sidebar ───────────────────────────────────────────── */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #08101e 0%, #05080f 100%) !important;
  border-right: 1px solid var(--border) !important;
  z-index: 999 !important;
}
section[data-testid="stSidebar"] .block-container{ padding-top: 2rem; }
button[data-testid="collapsedControl"],
button[kind="header"]{
  z-index: 9999 !important;
  color: var(--gold) !important;
}
.side-brand{
  display:flex; align-items:center; gap:10px; margin-bottom:18px;
  padding-bottom: 16px; border-bottom: 1px solid var(--border);
}
.side-brand .mark{
  width:34px; height:34px; border-radius:10px;
  background: conic-gradient(from 220deg, #d4af37, #b8860b, #f6e27a);
  display:grid; place-items:center; color:#0b1220; font-size:16px;
}
.side-brand .name{ font-family:'Cormorant Garamond',serif; font-size:18px; font-weight:700; color:var(--ink); line-height:1;}
.side-brand .name small{ display:block; font-family:'Plus Jakarta Sans',sans-serif; font-size:10px; letter-spacing:2px; color:var(--gold-soft); margin-top:4px; }
.history-item{
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 10px;
  cursor: pointer;
  transition: all .2s ease;
}
.history-item:hover{ border-color: var(--border-strong); background: rgba(212,175,55,0.04); }
.history-item .q{
  font-size: 13px; color: var(--ink); font-weight:500;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
  overflow:hidden; line-height:1.45;
}
.history-item .time{ font-size: 10px; color: var(--ink-mute); margin-top: 6px; letter-spacing: 1px; text-transform: uppercase; }
.history-empty{
  text-align:center; padding: 30px 14px; color:var(--ink-mute); font-size:13px;
  border: 1px dashed var(--border); border-radius: 12px;
}
.history-empty i{ font-size:24px; color: var(--gold); display:block; margin-bottom:10px; opacity:.6; }
section[data-testid="stSidebar"] .stButton > button{
  background: rgba(212,175,55,0.06) !important;
  border: 1px solid var(--border-strong) !important;
  color: var(--gold-soft) !important;
  border-radius: 10px !important;
  font-weight: 600 !important; font-size: 12px !important;
  letter-spacing: 1px !important; text-transform: uppercase !important;
  width: 100% !important;
}
section[data-testid="stSidebar"] .stButton > button:hover{
  background: rgba(212,175,55,0.12) !important;
  color: #f6e27a !important;
}

/* ── Stats Strip ───────────────────────────────────────── */
.stat-strip{
  display:grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
  margin: 8px 0 24px;
}
.stat{
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid var(--border);
  border-radius: 14px; padding: 16px 18px;
}
.stat .ico{
  width:34px; height:34px; border-radius:10px;
  display:grid; place-items:center; color: var(--gold);
  background: rgba(212,175,55,0.08); border:1px solid var(--border-strong);
  margin-bottom: 10px; font-size: 14px;
}
.stat .v{ font-family:'Cormorant Garamond',serif; font-size:24px; font-weight:700; color: var(--ink); line-height:1; }
.stat .l{ font-size:11px; color: var(--ink-mute); letter-spacing:2px; text-transform:uppercase; margin-top:6px;}

/* ── Loader ────────────────────────────────────────────── */
.thinking{
  display:flex; align-items:center; gap:14px;
  padding: 18px 22px; border-radius: 14px;
  background: rgba(212,175,55,0.05);
  border:1px solid var(--border-strong);
  color: var(--gold-soft); font-weight: 600; font-size: 14px;
  letter-spacing: .5px;
}
.thinking .dots{ display:inline-flex; gap:5px; }
.thinking .dots span{
  width:7px; height:7px; border-radius:50%; background: var(--gold);
  animation: bounce 1.2s infinite ease-in-out;
}
.thinking .dots span:nth-child(2){ animation-delay:.15s; }
.thinking .dots span:nth-child(3){ animation-delay:.3s; }
@keyframes bounce{ 0%,80%,100%{transform:translateY(0); opacity:.4;} 40%{transform:translateY(-6px); opacity:1;} }

/* ── Footer ────────────────────────────────────────────── */
.lex-footer{
  text-align:center; margin-top: 50px; padding: 22px 0 0;
  border-top: 1px solid var(--border);
  color: var(--ink-mute); font-size: 12px; letter-spacing: 1px;
}
.lex-footer .row{ display:flex; justify-content:center; gap: 18px; flex-wrap:wrap; margin-bottom:10px;}
.lex-footer .row span{ display:inline-flex; align-items:center; gap:6px; }
.lex-footer .row i{ color: var(--gold); }
.lex-footer .fineprint{ color: var(--ink-mute); font-size: 11px; max-width: 720px; margin: 8px auto 0; line-height:1.6; }

.stAlert{ border-radius: 14px !important; border: 1px solid var(--border-strong) !important; }

@media (max-width: 768px){
  .hero{ padding: 28px 22px; }
  .hero h1{ font-size: 36px; }
  .stat-strip{ grid-template-columns: 1fr; }
  .result-grid{ grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Session State ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""


# ─── PDF Generator ───────────────────────────────────────────────────────────
def generate_counsel_pdf(question: str, answer: str, sources: list) -> bytes:
    """Generate a clean PDF for AI Counsel answer."""
    try:
        from fpdf import FPDF

        class PDF(FPDF):
            def __init__(self):
                super().__init__()
                self.add_font("dejavusans", "", "fonts/DejaVuSans.ttf")
                self.add_font("dejavusans", "B", "fonts/DejaVuSans-Bold.ttf")
                self.add_font("dejavusans", "I", "fonts/DejaVuSans-Oblique.ttf")
                self.add_font("dejavusans", "BI", "fonts/DejaVuSans-BoldOblique.ttf")
                self.add_font("dejavusansmono", "", "fonts/DejaVuSansMono.ttf")
                self.add_font("dejavusansmono", "B", "fonts/DejaVuSansMono-Bold.ttf")
                self.add_font("dejavusansmono", "I", "fonts/DejaVuSansMono-Oblique.ttf")
                self.add_font("dejavusansmono", "BI", "fonts/DejaVuSansMono-BoldOblique.ttf")

            def header(self):
                self.set_fill_color(7, 11, 20)
                self.rect(0, 0, 210, 30, 'F')
                self.set_font("dejavusans", "B", 16)
                self.set_text_color(212, 175, 55)
                self.cell(0, 12, "LexTax PK", ln=False, align="L")
                self.set_font("dejavusans", "", 9)
                self.set_text_color(154, 167, 189)
                self.cell(0, 12, "PAKISTAN TAX INTELLIGENCE", ln=True, align="R")
                self.set_draw_color(212, 175, 55)
                self.set_line_width(0.5)
                self.line(10, 28, 200, 28)
                self.ln(4)

            def footer(self):
                self.set_y(-15)
                self.set_font("dejavusans", "I", 8)
                self.set_text_color(107, 119, 148)
                self.cell(0, 10,
                    f"LexTax PK  |  Generated {datetime.now().strftime('%d %b %Y, %H:%M')}  |  For reference only — not legal advice",
                    align="C"
                )

        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # ── Meta info
        pdf.set_font("dejavusans", style="B", size=12)
        pdf.set_text_color(107, 119, 148)
        pdf.cell(0, 6, f"Advisory Opinion  |  {datetime.now().strftime('%d %B %Y, %H:%M')}", ln=True)
        pdf.ln(3)

        # ── Question box
        pdf.set_fill_color(13, 20, 38)
        pdf.set_draw_color(212, 175, 55)
        pdf.set_line_width(0.3)
        pdf.rect(10, pdf.get_y(), 190, 8 + len(question) // 90 * 5 + 8, 'DF')
        pdf.set_font("dejavusans", style="B", size=9)
        pdf.set_text_color(232, 200, 112)
        pdf.set_x(14)
        pdf.cell(0, 6, "CONSULTATION QUESTION", ln=True)
        pdf.set_font("dejavusans", style="", size=11)
        pdf.set_text_color(232, 237, 245)
        pdf.set_x(14)
        pdf.multi_cell(182, 6, question)
        pdf.ln(6)

        # ── Divider
        pdf.set_draw_color(212, 175, 55)
        pdf.set_line_width(0.3)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # ── Answer heading
        pdf.set_font("dejavusans", style="B", size=12)
        pdf.set_text_color(232, 200, 112)
        pdf.cell(0, 6, "LEGAL COUNSEL  —  FBR-GROUNDED ADVISORY OPINION", ln=True)
        pdf.ln(2)

        # ── Answer body (clean text, strip markdown-ish chars)
        clean_answer = answer.replace("**", "").replace("__", "").replace("##", "").replace("#", "")
        pdf.set_font("dejavusans", style="B", size=12)
        pdf.set_text_color(232, 237, 245)
        pdf.set_fill_color(10, 18, 34)

        for line in clean_answer.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(3)
                continue
            if line.startswith("- ") or line.startswith("• "):
                pdf.set_x(16)
                pdf.set_text_color(212, 175, 55)
                pdf.cell(5, 6, "•")
                pdf.set_text_color(232, 237, 245)
                pdf.set_x(21)
                pdf.multi_cell(179, 6, line[2:])
            else:
                pdf.multi_cell(190, 6, line)

        # ── Sources
        if sources:
            pdf.ln(5)
            pdf.set_draw_color(212, 175, 55)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)
            pdf.set_font("dejavusans", style="B", size=12)
            pdf.set_text_color(232, 200, 112)
            pdf.cell(0, 6, "SOURCE CITATIONS  —  RETRIEVED FBR EXCERPTS", ln=True)
            pdf.ln(2)
            for i, src in enumerate(sources):
                pdf.set_font("dejavusans", style="B", size=12)
                pdf.set_text_color(212, 175, 55)
                pdf.cell(0, 5, f"Source {i+1}", ln=True)
                pdf.set_font("dejavusansmono", style="", size=9)
                pdf.set_text_color(154, 167, 189)
                pdf.set_fill_color(10, 18, 34)
                pdf.multi_cell(190, 5, src[:600] + ("..." if len(src) > 600 else ""), fill=True)
                pdf.ln(3)

        # ── Disclaimer
        pdf.ln(4)
        pdf.set_fill_color(8, 16, 30)
        pdf.set_draw_color(16, 185, 129)
        pdf.set_line_width(0.3)
        y = pdf.get_y()
        pdf.rect(10, y, 190, 14, 'DF')
        pdf.set_font("dejavusans", style="B", size=8)
        pdf.set_text_color(52, 211, 153)
        pdf.set_x(14)
        pdf.cell(0, 5, "DISCLAIMER", ln=True)
        pdf.set_font("dejavusans", style="", size=8)
        pdf.set_text_color(154, 167, 189)
        pdf.set_x(14)
        pdf.multi_cell(182, 4,
            "This document provides general informational guidance based on FBR documents. "
            "It does not constitute formal legal or tax advice. Consult a certified tax practitioner for binding decisions."
        )

        return bytes(pdf.output())

    except ImportError:
        return None


def generate_calc_pdf(r: dict, income_type: str) -> bytes:
    """Generate a PDF for Tax Calculator result."""
    try:
        from fpdf import FPDF

        class PDF(FPDF):
            def __init__(self):
                super().__init__()
                self.add_font("dejavusans", "", "fonts/DejaVuSans.ttf")
                self.add_font("dejavusans", "B", "fonts/DejaVuSans-Bold.ttf")
                self.add_font("dejavusans", "I", "fonts/DejaVuSans-Oblique.ttf")
                self.add_font("dejavusans", "BI", "fonts/DejaVuSans-BoldOblique.ttf")
                self.add_font("dejavusansmono", "", "fonts/DejaVuSansMono.ttf")
                self.add_font("dejavusansmono", "B", "fonts/DejaVuSansMono-Bold.ttf")
                self.add_font("dejavusansmono", "I", "fonts/DejaVuSansMono-Oblique.ttf")
                self.add_font("dejavusansmono", "BI", "fonts/DejaVuSansMono-BoldOblique.ttf")

            def header(self):
                self.set_fill_color(7, 11, 20)
                self.rect(0, 0, 210, 30, 'F')
                self.set_font("dejavusans", "B", 16)
                self.set_text_color(212, 175, 55)
                self.cell(0, 12, "LexTax PK", ln=False, align="L")
                self.set_font("dejavusans", "", 9)
                self.set_text_color(154, 167, 189)
                self.cell(0, 12, "TAX CALCULATOR — FBR 2024-25", ln=True, align="R")
                self.set_draw_color(212, 175, 55)
                self.set_line_width(0.5)
                self.line(10, 28, 200, 28)
                self.ln(4)

            def footer(self):
                self.set_y(-15)
                self.set_font("dejavusans", "I", 8)
                self.set_text_color(107, 119, 148)
                self.cell(0, 10,
                    f"LexTax PK  |  Generated {datetime.now().strftime('%d %b %Y, %H:%M')}  |  For reference only",
                    align="C"
                )

        def fmt_pdf(n):
            return f"PKR {n:,.0f}"

        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # Meta
        pdf.set_font("dejavusans", style="", size=9)
        pdf.set_text_color(107, 119, 148)
        pdf.cell(0, 6,
            f"Tax Year 2024-25  |  {datetime.now().strftime('%d %B %Y')}  |  "
            f"{'Active Filer' if r['filer'] else 'Non-Filer'}",
            ln=True
        )
        pdf.ln(4)

        # Title
        pdf.set_font("dejavusans", style="B", size=14)
        pdf.set_text_color(232, 237, 245)
        pdf.cell(0, 8, "Income Tax Computation — Salaried Person", ln=True)
        pdf.ln(2)

        # Input
        pdf.set_font("dejavusans", style="", size=10)
        pdf.set_text_color(154, 167, 189)
        pdf.cell(0, 6,
            f"Input: {income_type}  |  Annual Income: {fmt_pdf(r['annual_income'])}",
            ln=True
        )
        pdf.ln(6)

        # ── Summary table
        pdf.set_draw_color(212, 175, 55)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("dejavusans", style="B", size=9)
        pdf.set_text_color(232, 200, 112)
        pdf.cell(0, 6, "TAX SUMMARY", ln=True)
        pdf.ln(2)

        rows = [
            ("Annual Gross Income",    fmt_pdf(r["annual_income"]),  False),
            ("Annual Tax Payable",     fmt_pdf(r["annual_tax"]),     True),
            ("Monthly Tax Deduction",  fmt_pdf(r["monthly_tax"]),    True),
            ("Effective Tax Rate",     f"{r['effective_rate']:.2f}%",False),
            ("Net Annual Take-Home",   fmt_pdf(r["net_annual"]),     False),
            ("Net Monthly Take-Home",  fmt_pdf(r["net_monthly"]),    False),
        ]
        if not r["filer"] and r["surcharge"] > 0:
            rows.insert(2, ("Non-Filer Surcharge (25%)", fmt_pdf(r["surcharge"]), True))

        for label, value, is_tax in rows:
            pdf.set_fill_color(13, 20, 38) if not is_tax else pdf.set_fill_color(20, 10, 10)
            pdf.set_font("dejavusans", style="", size=10)
            pdf.set_text_color(154, 167, 189) if not is_tax else pdf.set_text_color(248, 113, 113)
            pdf.cell(120, 7, label, fill=True)
            pdf.set_font("dejavusans", style="B", size=10)
            pdf.set_text_color(232, 200, 112) if not is_tax else pdf.set_text_color(248, 113, 113)
            pdf.cell(70, 7, value, fill=True, ln=True)

        pdf.ln(6)

        # ── Slab table
        pdf.set_draw_color(212, 175, 55)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("dejavusans", style="B", size=12)
        pdf.set_text_color(232, 200, 112)
        pdf.cell(0, 6, "FBR SALARY TAX SLABS 2024-25", ln=True)
        pdf.ln(2)

        slab_labels = [
            "Up to PKR 600,000",
            "PKR 600,001 - 1,200,000",
            "PKR 1,200,001 - 2,200,000",
            "PKR 2,200,001 - 3,200,000",
            "PKR 3,200,001 - 4,100,000",
            "Above PKR 4,100,000",
        ]
        rates_list  = ["0%", "1%", "11%", "23%", "30%", "35%"]
        fixed_list  = ["—", "—", "PKR 6,000", "PKR 116,000", "PKR 346,000", "PKR 616,000"]

        # Header row
        pdf.set_fill_color(20, 26, 46)
        pdf.set_font("dejavusans", style="B", size=8)
        pdf.set_text_color(232, 200, 112)
        pdf.cell(90, 6, "Income Range", fill=True)
        pdf.cell(30, 6, "Rate", fill=True)
        pdf.cell(70, 6, "Fixed Amount", fill=True, ln=True)

        for i, (sl, rt, fx) in enumerate(zip(slab_labels, rates_list, fixed_list)):
            is_active = (i == r["active_slab"])
            if is_active:
                pdf.set_fill_color(30, 25, 5)
                pdf.set_text_color(232, 200, 112)
            else:
                pdf.set_fill_color(10, 16, 30)
                pdf.set_text_color(154, 167, 189)
            pdf.set_font("dejavusans", style="B" if is_active else "", size=8)
            marker = " << Your Slab" if is_active else ""
            pdf.cell(90, 6, sl + marker, fill=True)
            pdf.cell(30, 6, rt, fill=True)
            pdf.cell(70, 6, fx, fill=True, ln=True)

        # Filer tip
        if not r["filer"]:
            filer_r = calculate_salary_tax(r["annual_income"], True)
            saving  = r["annual_tax"] - filer_r["annual_tax"]
            pdf.ln(5)
            pdf.set_fill_color(5, 20, 15)
            pdf.set_draw_color(16, 185, 129)
            pdf.set_line_width(0.3)
            y2 = pdf.get_y()
            pdf.rect(10, y2, 190, 12, 'DF')
            pdf.set_font("dejavusans", style="B", size=9)
            pdf.set_text_color(52, 211, 153)
            pdf.set_x(14)
            pdf.cell(0, 5, "TIP: Become an Active Filer", ln=True)
            pdf.set_font("dejavusans", style="", size=9)
            pdf.set_x(14)
            pdf.multi_cell(182, 4,
                f"Register as a tax filer and save {fmt_pdf(saving)} per year in taxes!"
            )

        # Disclaimer
        pdf.ln(6)
        pdf.set_fill_color(8, 16, 30)
        pdf.set_draw_color(16, 185, 129)
        y3 = pdf.get_y()
        pdf.rect(10, y3, 190, 12, 'DF')
        pdf.set_font("dejavusans", style="B", size=8)
        pdf.set_text_color(52, 211, 153)
        pdf.set_x(14)
        pdf.cell(0, 5, "DISCLAIMER", ln=True)
        pdf.set_font("dejavusans", style="", size=8)
        pdf.set_text_color(154, 167, 189)
        pdf.set_x(14)
        pdf.multi_cell(182, 4,
            "Tax calculations are based on FBR Finance Act 2024 salaried slabs and are for reference only. "
            "Consult a certified tax practitioner for official filing."
        )

        return bytes(pdf.output())

    except ImportError:
        return None


# ─── Tax Calculator Logic (FBR 2024-25 Slabs) ────────────────────────────────
def calculate_salary_tax(annual_income: float, filer: bool) -> dict:
    slabs = [
        (0,         600_000,     0.00,  0),
        (600_001,   1_200_000,   0.01,  600_000),
        (1_200_001, 2_200_000,   0.11,  1_200_000),
        (2_200_001, 3_200_000,   0.23,  2_200_000),
        (3_200_001, 4_100_000,   0.30,  3_200_000),
        (4_100_001, float("inf"),0.35,  4_100_000),
    ]
    slab_fixed_tax = [0, 0, 6_000, 116_000, 346_000, 616_000]

    tax = 0.0
    active_slab_index = 0

    for i, (low, high, rate, base) in enumerate(slabs):
        if annual_income > low:
            active_slab_index = i
            if annual_income <= high:
                tax = slab_fixed_tax[i] + (annual_income - base) * rate
                break
            elif i == len(slabs) - 1:
                tax = slab_fixed_tax[i] + (annual_income - base) * rate

    surcharge = 0.0
    if not filer and tax > 0:
        surcharge = tax * 0.25
        tax += surcharge

    monthly_tax    = tax / 12
    net_annual     = annual_income - tax
    net_monthly    = net_annual / 12
    effective_rate = (tax / annual_income * 100) if annual_income > 0 else 0

    return {
        "annual_income":  annual_income,
        "annual_tax":     tax,
        "surcharge":      surcharge,
        "monthly_tax":    monthly_tax,
        "net_annual":     net_annual,
        "net_monthly":    net_monthly,
        "effective_rate": effective_rate,
        "active_slab":    active_slab_index,
        "filer":          filer,
    }


def fmt(n: float) -> str:
    return f"PKR {n:,.0f}"


# ─── Vector DB & LLM ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_db():
    # BGE-small gives noticeably better retrieval accuracy than MiniLM for
    # structured/numeric text (tax slabs, rates, ranges) at a similar speed/size.
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    salary_loader   = TextLoader("knowledge_base/salary_slabs.txt")
    filer_loader    = TextLoader("knowledge_base/filer_benefits.txt")
    ordinance_loader = TextLoader("knowledge_base/fbr_heavy_tax_ordinance.txt")
    all_documents = salary_loader.load() + filer_loader.load() + ordinance_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    return db


@st.cache_resource(show_spinner=False)
def load_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model="openrouter/free",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
        default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "LexTax PK"},
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="mark"><i class="fa-solid fa-scale-balanced"></i></div>
            <div class="name">LexTax PK<small>CONSULTATION LEDGER</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        if st.button("Clear History", key="clear_hist"):
            st.session_state.history = []
            st.rerun()

    st.markdown(
        '<div class="section-label" style="margin-top:18px;">Recent Queries</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.markdown(
            """
            <div class="history-empty">
                <i class="fa-regular fa-folder-open"></i>
                No consultations yet.<br/>Ask your first question to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for idx, item in enumerate(reversed(st.session_state.history[-12:])):
            st.markdown(
                f"""
                <div class="history-item">
                    <div class="q">{item['q']}</div>
                    <div class="time"><i class="fa-regular fa-clock"></i>&nbsp; {item['time']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div style="margin-top:30px; padding:14px; border-radius:12px;
             background: rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.2);
             font-size:11.5px; color:#9aa7bd; line-height:1.6;">
            <div style="color:#34d399; font-weight:700; letter-spacing:2px;
                 text-transform:uppercase; font-size:10px; margin-bottom:6px;">
                <i class="fa-solid fa-shield-halved"></i>&nbsp; Disclaimer
            </div>
            LexTax PK provides informational guidance based on FBR documents.
            For binding legal decisions, consult a certified tax practitioner.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Brand Bar ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="brandbar">
        <div class="brand">
            <div class="brand-mark"><i class="fa-solid fa-scale-balanced"></i></div>
            <div class="brand-title">LexTax&nbsp;PK<small>PAKISTAN TAX INTELLIGENCE</small></div>
        </div>
        <div class="brand-pills">
            <span class="pill"><i class="fa-solid fa-landmark"></i>&nbsp; FBR Aligned</span>
            <span class="pill live">Live Advisor</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <span class="hero-eyebrow"><i class="fa-solid fa-gavel"></i>&nbsp; Trusted Tax Counsel</span>
        <h1>Pakistan&rsquo;s <span class="accent">AI&nbsp;Tax&nbsp;Counsel</span><br/>at your fingertips.</h1>
        <p>Get clear, document-grounded answers on FBR salary slabs, filer vs non-filer benefits,
        withholding taxes and more &mdash; powered by Retrieval-Augmented Generation across the latest tax circulars.</p>
        <div class="hero-meta">
            <div class="meta-chip"><i class="fa-solid fa-book-open-reader"></i>&nbsp; Cites Official Sources</div>
            <div class="meta-chip"><i class="fa-solid fa-language"></i>&nbsp; English &amp; Roman Urdu</div>
            <div class="meta-chip"><i class="fa-solid fa-bolt"></i>&nbsp; Instant Answers</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Stats Strip ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="stat-strip">
        <div class="stat">
            <div class="ico"><i class="fa-solid fa-file-shield"></i></div>
            <div class="v">FBR&nbsp;Circulars</div>
            <div class="l">Indexed Sources</div>
        </div>
        <div class="stat">
            <div class="ico"><i class="fa-solid fa-brain"></i></div>
            <div class="v">RAG&nbsp;+&nbsp;LLM</div>
            <div class="l">Retrieval Engine</div>
        </div>
        <div class="stat">
            <div class="ico"><i class="fa-solid fa-calculator"></i></div>
            <div class="v">Tax&nbsp;Calc</div>
            <div class="l">FBR 2024-25 Slabs</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_counsel, tab_calc = st.tabs(["⚖️  AI Tax Counsel", "🧮  Tax Calculator"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — AI COUNSEL
# ════════════════════════════════════════════════════════════════════════════
with tab_counsel:
    try:
        vector_db = load_db()
        llm = load_llm()

        if llm is None:
            st.error("OPENROUTER_API_KEY missing. Please add it to your .env file.")
            st.stop()

        st.markdown(
            '<div class="section-label"><i class="fa-solid fa-wand-magic-sparkles"></i>&nbsp; Suggested Consultations</div>',
            unsafe_allow_html=True,
        )
        sample_questions = [
            ("Salary Tax Slab",  "What is the tax on PKR 800,000 annual salary?"),
            ("Filer Advantage",  "Filer vs Non-Filer — what benefits do I get as a filer?"),
            ("High Earners",     "Tax computation for 2,000,000 PKR annual salary?"),
            ("Withholding",      "Explain withholding tax on bank transactions for non-filers."),
            ("Return Filing",    "Step-by-step process to become an active tax filer in Pakistan?"),
            ("Property Tax",     "Capital gains tax on property sale in Pakistan?"),
        ]
        st.markdown('<div class="suggest-grid">', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (label, q) in enumerate(sample_questions):
            with cols[i % 3]:
                if st.button(f"⚖  {label}\n{q}", key=f"sample_{i}", use_container_width=True):
                    st.session_state.prefill = q
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-label" style="margin-top:28px;"><i class="fa-solid fa-comments"></i>&nbsp; Your Consultation</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ask-shell">', unsafe_allow_html=True)
        with st.form("ask_form", clear_on_submit=False):
            user_query = st.text_area(
                "Question",
                value=st.session_state.prefill,
                placeholder="e.g., What is the income tax on PKR 1,200,000 annual salary for a filer in Pakistan?",
                label_visibility="collapsed",
                height=90,
            )
            col_a, col_b = st.columns([3, 1])
            with col_b:
                submitted = st.form_submit_button("⚖  Seek Counsel", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.prefill:
            st.session_state.prefill = ""

        if submitted:
            if not user_query.strip():
                st.warning("Please enter a question before requesting counsel.")
            else:
                loader_ph = st.empty()
                loader_ph.markdown(
                    """
                    <div class="thinking">
                        <i class="fa-solid fa-scroll"></i>
                        Reviewing FBR circulars &amp; drafting your counsel
                        <span class="dots"><span></span><span></span><span></span></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # BGE models retrieve better with an instruction-style query prefix
                search_query = f"Represent this question for retrieving relevant tax information: {user_query}"
                docs    = vector_db.similarity_search(search_query, k=8)
                context = "\n\n".join([d.page_content for d in docs])
                prompt  = f"""You are LexTax PK, a senior FBR Tax Advisor for Pakistan.
Answer the user's question strictly based on the context below.
If the answer is not in the context, say: "I don't have enough FBR data on this."
Keep the answer clear, structured, and helpful with bullet points where needed.
Respond in the same language as the user's question (English or Roman Urdu).

Context:
{context}

User Question: {user_query}

Answer:"""
                response      = llm.invoke(prompt)
                response_text = response.content
                loader_ph.empty()

                st.session_state.history.append({
                    "q":    user_query,
                    "a":    response_text,
                    "time": datetime.now().strftime("%d %b · %H:%M"),
                })

                # ── Answer card
                safe_answer = response_text.replace("\n", "<br>")
                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="answer-head">
                            <div class="left">
                                <div class="seal"><i class="fa-solid fa-scale-balanced"></i></div>
                                <div>
                                    <div class="answer-title">Legal Counsel</div>
                                    <div class="answer-sub">LexTax PK · Advisory Opinion</div>
                                </div>
                            </div>
                            <span class="answer-tag"><i class="fa-solid fa-check-double"></i>&nbsp; Grounded in FBR Docs</span>
                        </div>
                        <div class="answer-body">{safe_answer}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── PDF Download Button
                source_texts = [d.page_content for d in docs]
                pdf_bytes = generate_counsel_pdf(user_query, response_text, source_texts)

                if pdf_bytes:
                    fname = f"LexTaxPK_Counsel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.markdown('<div class="pdf-btn">', unsafe_allow_html=True)
                    st.download_button(
                        label="📄  Download Advisory Opinion as PDF",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf",
                        key=f"dl_counsel_{datetime.now().timestamp()}",
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("PDF download requires `fpdf2`. Run: `pip install fpdf2`")

                # ── Sources expander
                with st.expander("📜  View Source Citations  ·  Retrieved FBR Excerpts", expanded=False):
                    for i, doc in enumerate(docs):
                        st.markdown(
                            f"""
                            <div class="context-box">
                                <span class="src-label"><i class="fa-solid fa-bookmark"></i>&nbsp; Source {i+1}</span><br/>
                                {doc.page_content}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.info(
            "Quick checklist · (1) `chroma_db` folder exists · "
            "(2) `OPENROUTER_API_KEY` is set in your `.env` file"
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — TAX CALCULATOR
# ════════════════════════════════════════════════════════════════════════════
with tab_calc:
    st.markdown(
        '<div class="section-label"><i class="fa-solid fa-calculator"></i>&nbsp; Pakistan Income Tax Calculator — FBR 2024-25</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="calc-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])

    with col1:
        income_type = st.selectbox(
            "Income Type",
            ["Monthly Salary", "Annual Salary"],
            key="calc_income_type",
        )
        income_val = st.number_input(
            f"Enter {income_type} (PKR)",
            min_value=0,
            max_value=100_000_000,
            value=100_000,
            step=10_000,
            key="calc_income_val",
            help="Enter your gross salary before any deductions.",
        )

    with col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        filer_status = st.radio(
            "Tax Filer Status",
            ["Active Filer", "Non-Filer"],
            key="calc_filer",
            help="Non-filers pay 25% extra surcharge on their tax.",
        )
        is_filer = filer_status == "Active Filer"

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="calc-cta">', unsafe_allow_html=True)
    calc_btn = st.button("🧮  Calculate My Tax", key="calc_submit", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if calc_btn:
        annual = float(income_val) * 12 if income_type == "Monthly Salary" else float(income_val)
        r = calculate_salary_tax(annual, is_filer)

        st.markdown(
            f"""<div class="calc-result">
<div class="answer-head">
<div class="left">
<div class="seal"><i class="fa-solid fa-calculator"></i></div>
<div>
<div class="answer-title">Tax Breakdown</div>
<div class="answer-sub">FBR Salaried Person · Tax Year 2024-25</div>
</div>
</div>
<span class="answer-tag">
<i class="fa-solid fa-{'user-check' if r['filer'] else 'user-xmark'}"></i>&nbsp;
{'Active Filer' if r['filer'] else 'Non-Filer'}
</span>
</div>
<div class="result-grid">
<div class="result-box"><div class="rb-label">Annual Income</div><div class="rb-value">{fmt(r['annual_income'])}</div></div>
<div class="result-box"><div class="rb-label">Annual Tax</div><div class="rb-value red">{fmt(r['annual_tax'])}</div></div>
<div class="result-box"><div class="rb-label">Effective Rate</div><div class="rb-value">{r['effective_rate']:.2f}%</div></div>
<div class="result-box"><div class="rb-label">Monthly Tax</div><div class="rb-value red">{fmt(r['monthly_tax'])}</div></div>
<div class="result-box"><div class="rb-label">Net Monthly Take-Home</div><div class="rb-value green">{fmt(r['net_monthly'])}</div></div>
<div class="result-box"><div class="rb-label">Net Annual Take-Home</div><div class="rb-value green">{fmt(r['net_annual'])}</div></div>
</div>""",
            unsafe_allow_html=True,
        )

        if not r["filer"] and r["surcharge"] > 0:
            st.markdown(
                f"""
                <div style="margin: 8px 0 16px; padding: 12px 16px; border-radius: 12px;
                     background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.3);
                     color: #fca5a5; font-size: 13px;">
                    <i class="fa-solid fa-triangle-exclamation"></i>&nbsp;
                    <strong>Non-Filer Surcharge:</strong> {fmt(r['surcharge'])} extra tax applied (25% penalty).
                    Become a filer to save this amount!
                </div>
                """,
                unsafe_allow_html=True,
            )

        slab_labels = [
            "Up to PKR 600,000",
            "PKR 600,001 – 1,200,000",
            "PKR 1,200,001 – 2,200,000",
            "PKR 2,200,001 – 3,200,000",
            "PKR 3,200,001 – 4,100,000",
            "Above PKR 4,100,000",
        ]
        rates_disp = ["0%", "1%", "11%", "23%", "30%", "35%"]
        fixed_disp = ["—", "—", "PKR 6,000", "PKR 116,000", "PKR 346,000", "PKR 616,000"]

        rows_html = ""
        for i, (label, rate, fix) in enumerate(zip(slab_labels, rates_disp, fixed_disp)):
            active = "active-slab" if i == r["active_slab"] else ""
            marker = " ◀ Your Slab" if i == r["active_slab"] else ""
            rows_html += f'<tr class="{active}"><td>{label}{marker}</td><td>{rate}</td><td>{fix}</td></tr>'

        st.markdown(
            f"""
            <div style="margin-top: 16px;">
                <div style="font-size:11px; font-weight:700; letter-spacing:2px;
                     text-transform:uppercase; color:var(--gold-soft); margin-bottom:10px;">
                    <i class="fa-solid fa-table"></i>&nbsp; FBR Salary Tax Slabs 2024-25
                </div>
                <table class="slab-table">
                    <thead><tr><th>Income Range</th><th>Tax Rate</th><th>Fixed Amount</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not r["filer"]:
            filer_result = calculate_salary_tax(annual, True)
            saving = r["annual_tax"] - filer_result["annual_tax"]
            st.markdown(
                f"""
                <div style="margin-top:16px; padding: 14px 18px; border-radius: 14px;
                     background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25);
                     color: var(--emerald-soft); font-size: 13.5px;">
                    <i class="fa-solid fa-lightbulb"></i>&nbsp;
                    <strong>Tip:</strong> Become an active filer and save
                    <strong>{fmt(saving)}</strong> per year!
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── PDF Download for Calculator
        pdf_calc = generate_calc_pdf(r, income_type)
        if pdf_calc:
            fname_c = f"LexTaxPK_TaxCalc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.markdown('<div class="pdf-btn" style="margin-top:18px;">', unsafe_allow_html=True)
            st.download_button(
                label="📄  Download Tax Breakdown as PDF",
                data=pdf_calc,
                file_name=fname_c,
                mime="application/pdf",
                key=f"dl_calc_{datetime.now().timestamp()}",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("PDF download requires `fpdf2`. Run: `pip install fpdf2`")

    else:
        st.markdown(
            """
            <div style="margin-top: 24px; padding: 20px 24px; border-radius: 16px;
                 background: rgba(255,255,255,0.02); border: 1px solid var(--border);">
                <div style="font-size:11px; font-weight:700; letter-spacing:2px;
                     text-transform:uppercase; color:var(--gold-soft); margin-bottom:14px;">
                    <i class="fa-solid fa-info-circle"></i>&nbsp; FBR Salaried Tax Slabs 2024-25 (Quick Reference)
                </div>
                <table class="slab-table">
                    <thead><tr><th>Annual Income</th><th>Tax Rate</th><th>Fixed Amount</th></tr></thead>
                    <tbody>
                        <tr><td>Up to PKR 600,000</td><td>0%</td><td>—</td></tr>
                        <tr><td>PKR 600,001 – 1,200,000</td><td>1%</td><td>—</td></tr>
                        <tr><td>PKR 1,200,001 – 2,200,000</td><td>11%</td><td>PKR 6,000</td></tr>
                        <tr><td>PKR 2,200,001 – 3,200,000</td><td>23%</td><td>PKR 116,000</td></tr>
                        <tr><td>PKR 3,200,001 – 4,100,000</td><td>30%</td><td>PKR 346,000</td></tr>
                        <tr><td>Above PKR 4,100,000</td><td>35%</td><td>PKR 616,000</td></tr>
                    </tbody>
                </table>
                <div style="margin-top:12px; font-size:12px; color:var(--ink-mute);">
                    <i class="fa-solid fa-triangle-exclamation" style="color:var(--gold)"></i>&nbsp;
                    Non-filers pay an additional 25% surcharge on the calculated tax amount.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="lex-footer">
        <div class="row">
            <span><i class="fa-solid fa-scale-balanced"></i> LexTax&nbsp;PK</span>
            <span><i class="fa-solid fa-database"></i> ChromaDB</span>
            <span><i class="fa-solid fa-link"></i> LangChain</span>
            <span><i class="fa-solid fa-bolt"></i> OpenRouter</span>
            <span><i class="fa-solid fa-calculator"></i> FBR&nbsp;2024-25</span>
        </div>
        <div class="fineprint">
            &copy; {datetime.now().year} LexTax&nbsp;PK &mdash; AI-assisted tax consultation grounded in Federal Board of Revenue (FBR) documents.
            Tax calculations are based on FBR Finance Act 2024 salaried slabs and are for reference only.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
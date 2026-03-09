import streamlit as st
import requests
import json
import re
import pandas as pd
import numpy as np
from data_agent import load_dataset, safe_exec, format_result, get_dataset_summary
from knowledge_base import ROUTER_SYSTEM_PROMPT

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Project — AI Assistant",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&family=Fira+Code&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .stApp { background: linear-gradient(135deg,#0f0c29 0%,#1a1040 40%,#24243e 100%); min-height:100vh; }

  [data-testid="stSidebar"] { background:rgba(255,255,255,0.04)!important; border-right:1px solid rgba(255,255,255,0.08); backdrop-filter:blur(12px); }
  [data-testid="stSidebar"] * { color:#e0d6ff!important; }

  /* gate */
  .gate-wrap { max-width:480px; margin:6vh auto 0; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.12); border-radius:20px; padding:2.5rem 2rem; text-align:center; backdrop-filter:blur(14px); }
  .gate-title { font-family:'DM Serif Display',serif; font-size:1.7rem; background:linear-gradient(90deg,#ff6b6b,#ee5a24); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0 0 0.4rem; }
  .gate-sub { color:#a89dcc; font-size:0.88rem; margin-bottom:1.4rem; line-height:1.5; }

  /* header */
  .hero-title { font-family:'DM Serif Display',serif; font-size:clamp(2rem,5vw,3.2rem); background:linear-gradient(90deg,#ff6b6b,#ee5a24,#ff6b6b); background-size:200% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent; animation:shimmer 3s linear infinite; margin:0; }
  @keyframes shimmer { 0%{background-position:0% center} 100%{background-position:200% center} }
  .hero-sub { color:#a89dcc; font-size:0.95rem; font-weight:300; margin-top:0.4rem; }

  /* bubbles */
  .msg-user { display:flex; justify-content:flex-end; margin:0.7rem 0; animation:fadeUp 0.3s ease; }
  @keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  .bubble-user { background:linear-gradient(135deg,#ee5a24,#d63031); color:#fff; border-radius:20px 20px 4px 20px; padding:0.75rem 1.1rem; max-width:70%; font-size:0.92rem; line-height:1.55; box-shadow:0 4px 18px rgba(238,90,36,0.35); }

  /* data result box */
  .data-result-box { background:rgba(46,213,115,0.06); border:1px solid rgba(46,213,115,0.25); border-radius:12px; padding:0.8rem 1rem; margin:0.5rem 0; }
  .data-badge { background:rgba(46,213,115,0.15); border:1px solid rgba(46,213,115,0.4); color:#2ed573; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:600; display:inline-block; margin-bottom:0.5rem; }
  .code-badge { background:rgba(108,92,231,0.15); border:1px solid rgba(108,92,231,0.4); color:#a29bfe; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:600; display:inline-block; margin-bottom:0.5rem; }

  /* stat cards */
  .stat-card { background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:0.9rem; text-align:center; margin-bottom:0.6rem; }
  .stat-val { font-size:1.5rem; font-weight:600; color:#ff6b6b; }
  .stat-lbl { font-size:0.75rem; color:#a89dcc; margin-top:2px; }

  .key-badge { background:rgba(46,213,115,0.15); border:1px solid rgba(46,213,115,0.4); color:#2ed573; border-radius:10px; padding:6px 12px; font-size:0.8rem; text-align:center; margin-bottom:0.5rem; }

  /* chat input */
  .stChatInput > div { background:rgba(255,255,255,0.06)!important; border:1px solid rgba(255,255,255,0.15)!important; border-radius:16px!important; }
  .stChatInput textarea { color:#fff!important; }
  .stChatInput button { color:#ee5a24!important; }

  hr { border-color:rgba(255,255,255,0.08)!important; }
  ::-webkit-scrollbar { width:5px; }
  ::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.15); border-radius:10px; }

  /* table styling inside chat */
  table { border-collapse:collapse; width:100%; font-size:0.82rem; }
  th { background:rgba(108,92,231,0.3); color:#d4cbff; padding:6px 10px; text-align:left; }
  td { padding:5px 10px; border-bottom:1px solid rgba(255,255,255,0.06); color:#e0d6ff; }
  tr:hover td { background:rgba(255,255,255,0.04); }
</style>
""", unsafe_allow_html=True)


# ── Groq API ──────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"

def call_groq_router(user_message: str, api_key: str) -> dict:
    """
    Ask Groq to route the message — returns parsed JSON dict.
    type: 'data_query' | 'knowledge' | 'off_topic'
    """
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},
            json=payload, timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # Extract JSON from response (handle markdown code fences)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"type": "knowledge", "answer": raw}

    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        if code == 401:
            return {"type": "error", "message": "❌ Invalid API key. Please check your Groq key."}
        return {"type": "error", "message": f"⚠️ Groq API error ({code}). Please try again."}
    except json.JSONDecodeError:
        return {"type": "knowledge", "answer": raw}
    except Exception as e:
        return {"type": "error", "message": f"⚠️ Error: {str(e)}"}


def call_groq_followup(question: str, data_result: str, api_key: str) -> str:
    """After getting data result, ask Groq to interpret it in plain English."""
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are a data analyst assistant for a Heart Disease Prediction ML project. "
                "The user asked a question, we ran a pandas query on the dataset, and got a result. "
                "Interpret the result in 1-3 clear, friendly sentences. Be direct and specific. "
                "No need to repeat the raw data if it's already shown — just explain what it means."
            )},
            {"role": "user", "content": f"Question: {question}\n\nData result:\n{data_result}"},
        ],
        "max_tokens": 300,
        "temperature": 0.3,
    }
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},
            json=payload, timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("pending_question", None),
    ("groq_api_key", ""),
    ("api_verified", False),
    ("df", None),
    ("show_code", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# API KEY GATE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.api_verified:
    st.markdown("""
    <div class="gate-wrap">
      <div style="font-size:3rem">🔑</div>
      <h2 class="gate-title">Enter Your Groq API Key</h2>
      <p class="gate-sub">
        Powered by <strong>Groq (free &amp; fast)</strong>.<br>
        Get your key at <a href="https://console.groq.com/keys" target="_blank" style="color:#ff6b6b;">console.groq.com/keys</a><br>
        Your key is only stored in this browser session.
      </p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...", label_visibility="collapsed")
        b1, b2 = st.columns(2)
        with b1:
            start_btn = st.button("🚀 Start Chatting", use_container_width=True, type="primary")
        with b2:
            help_btn  = st.button("📖 Get a key", use_container_width=True)

        if help_btn:
            st.info("1. Visit [console.groq.com/keys](https://console.groq.com/keys)\n2. Sign up (free)\n3. Click **Create API Key**\n4. Copy the `gsk_...` key and paste above")

        if start_btn:
            key = key_input.strip()
            if not key:
                st.error("Please enter your API key.")
            else:
                with st.spinner("Verifying…"):
                    test = call_groq_router("hello", key)
                if test.get("type") == "error" and "Invalid" in test.get("message",""):
                    st.error("Invalid API key. Please check and try again.")
                else:
                    st.session_state.groq_api_key = key
                    st.session_state.api_verified  = True
                    st.rerun()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATASET
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.df is None:
    df = load_dataset("heart.csv")
    st.session_state.df = df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ Project Info")
    masked = st.session_state.groq_api_key[:7] + "..." + st.session_state.groq_api_key[-4:]
    st.markdown(f'<div class="key-badge">🔑 Connected: {masked}</div>', unsafe_allow_html=True)
    if st.button("🔄 Change API Key", use_container_width=True):
        st.session_state.api_verified = False
        st.session_state.groq_api_key = ""
        st.session_state.messages     = []
        st.rerun()

    st.markdown("---")

    # Dataset status
    df = st.session_state.df
    if df is not None:
        st.markdown('<div class="key-badge" style="color:#74b9ff;border-color:rgba(116,185,255,0.4);background:rgba(116,185,255,0.1);">📊 heart.csv loaded ✓</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{len(df)}</div><div class="stat-lbl">Samples</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-card"><div class="stat-val">{df["target"].sum()}</div><div class="stat-lbl">With Disease</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{len(df.columns)-1}</div><div class="stat-lbl">Features</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-card"><div class="stat-val">{(df["target"]==0).sum()}</div><div class="stat-lbl">No Disease</div></div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ heart.csv not found.\nPlace heart.csv in the same folder as app.py to enable live data queries.")

    st.markdown("---")
    st.markdown("**🏆 Best Model**")
    st.markdown("Logistic Regression + Optimal Threshold")
    st.markdown("Precision: `0.8788` | Recall: `0.9062`")

    st.markdown("---")
    st.markdown("**🔗 Quick Links**")
    st.markdown("[📂 GitHub Repo](https://github.com/kanna-vamshi-krishna/heart-disease-prediction)")
    st.markdown("[📊 Kaggle Dataset](https://www.kaggle.com/datasets)")

    st.markdown("---")
    st.markdown("**⚡ Groq**")
    st.caption(f"Model: `{GROQ_MODEL}`")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()
    st.markdown("---")
    st.caption("Built by Kanna Vamshi Krishna\nChatbot — Groq LLaMA 3.3 + Streamlit")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.5rem 1rem 0.5rem">
  <h1 class="hero-title">❤️ Heart Disease Prediction</h1>
  <p class="hero-sub">Ask anything — I'll query the live dataset or explain the project</p>
</div>""", unsafe_allow_html=True)
st.markdown("---")


# ── Suggested questions ───────────────────────────────────────────────────────
SUGGESTED = [
    "How many have heart disease?",
    "Average age of patients?",
    "Show patients older than 65",
    "Male vs female count?",
    "Which model performed best?",
    "Avg cholesterol by disease?",
    "Explain ROC curve",
    "Top 5 highest cholesterol?",
]
st.markdown("**💡 Try asking:**")
cols = st.columns(4)
for i, q in enumerate(SUGGESTED):
    with cols[i % 4]:
        if st.button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()
st.markdown("---")


# ── Message renderer ──────────────────────────────────────────────────────────
def render_message(msg: dict, idx: int):
    role = msg["role"]
    if role == "user":
        st.markdown(f"""
        <div class="msg-user">
          <div class="bubble-user">{msg["content"]}</div>
          <div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#ee5a24,#d63031);display:flex;align-items:center;justify-content:center;margin-left:8px;flex-shrink:0">👤</div>
        </div>""", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="❤️"):
            msg_type = msg.get("msg_type", "knowledge")

            if msg_type == "data_query":
                # Show data badge + explanation
                st.markdown(f'<span class="data-badge">📊 LIVE DATA QUERY</span>', unsafe_allow_html=True)
                if msg.get("explanation"):
                    st.markdown(f"*{msg['explanation']}*")

                # Show result
                result_text = msg.get("result", "")
                if result_text:
                    st.markdown(result_text)

                # Show interpretation
                if msg.get("interpretation"):
                    st.info(f"💡 {msg['interpretation']}")

                # Toggle code
                code = msg.get("code", "")
                if code:
                    show_key = f"show_code_{idx}"
                    if st.button(f"{'🙈 Hide' if st.session_state.show_code.get(show_key) else '👁️ Show'} pandas code", key=f"btn_code_{idx}"):
                        st.session_state.show_code[show_key] = not st.session_state.show_code.get(show_key, False)
                    if st.session_state.show_code.get(show_key):
                        st.markdown(f'<span class="code-badge">🐼 PANDAS CODE</span>', unsafe_allow_html=True)
                        st.code(code, language="python")

            elif msg_type == "off_topic":
                st.markdown("I'm designed specifically to answer questions about the **Heart Disease Prediction project** by Kanna Vamshi Krishna. I can't help with other topics — but feel free to ask me anything about this project! 😊")

            elif msg_type == "error":
                st.error(msg.get("content", "An error occurred."))

            else:  # knowledge
                st.markdown(msg["content"])


# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:2rem;color:#a89dcc;">
      <div style="font-size:3rem">🤖</div>
      <p style="margin-top:0.5rem;font-size:1rem;">
        Hi! I can answer questions about the project <em>and</em> run <strong style="color:#2ed573">live queries</strong> on the dataset.<br>
        Try: <em>"How many patients are older than 60?"</em> or <em>"Show average cholesterol by disease"</em>
      </p>
    </div>""", unsafe_allow_html=True)
else:
    for i, msg in enumerate(st.session_state.messages):
        render_message(msg, i)


# ── Core handler ──────────────────────────────────────────────────────────────
def handle_question(user_input: str):
    st.session_state.messages.append({"role": "user", "content": user_input})

    df = st.session_state.df
    api_key = st.session_state.groq_api_key

    with st.spinner("Thinking…"):
        routed = call_groq_router(user_input, api_key)

    rtype = routed.get("type", "knowledge")

    if rtype == "data_query":
        if df is None:
            st.session_state.messages.append({
                "role": "assistant",
                "msg_type": "error",
                "content": "⚠️ **heart.csv not found.** Please place `heart.csv` in the same folder as `app.py` and restart the app.",
            })
        else:
            code = routed.get("code", "")
            explanation = routed.get("explanation", "")
            exec_result = safe_exec(code, df)
            formatted = format_result(exec_result)

            # Get plain-English interpretation
            interpretation = ""
            if exec_result["success"]:
                with st.spinner("Interpreting result…"):
                    interpretation = call_groq_followup(user_input, formatted, api_key)

            st.session_state.messages.append({
                "role": "assistant",
                "msg_type": "data_query",
                "explanation": explanation,
                "code": code,
                "result": formatted,
                "interpretation": interpretation,
            })

    elif rtype == "knowledge":
        st.session_state.messages.append({
            "role": "assistant",
            "msg_type": "knowledge",
            "content": routed.get("answer", "I couldn't generate a response."),
        })

    elif rtype == "off_topic":
        st.session_state.messages.append({
            "role": "assistant",
            "msg_type": "off_topic",
            "content": "",
        })

    else:  # error
        st.session_state.messages.append({
            "role": "assistant",
            "msg_type": "error",
            "content": routed.get("message", "An unexpected error occurred."),
        })

    st.rerun()


# ── Suggested question handler ────────────────────────────────────────────────
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    handle_question(q)

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about the project or query the dataset live..."):
    handle_question(prompt)

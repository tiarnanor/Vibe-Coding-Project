# helios_lida_frontend.py
from __future__ import annotations

import io
import os
import json
import base64
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# Use headless backend for server-side image saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ LiteLLM / OpenAI proxy ============
#OPENAI_API_KEY = "ENTER YOUR API-KEY HERE"

#os.environ["OPENAI_API_KEY"]  = OPENAI_API_KEY

# Streamlit Cloud injects OPENAI_API_KEY automatically
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found. Check Streamlit Secrets.")


from openai import OpenAI
from llmx import llm
from lida import Manager, TextGenerationConfig

oa_client = OpenAI()
DEFAULT_MODEL = "gpt-4o"

# LIDA with llmx-backed OpenAI 
text_gen = llm(provider="openai", model=DEFAULT_MODEL)
lida     = Manager(text_gen=text_gen)

# ================= Page / styles =================
st.set_page_config(page_title="Helios AI (LIDA)", page_icon="📈", layout="wide")
st.markdown("""
<style>
  .stApp { background:#f5f7fb; color:#1b1f23; }
  h1 { text-align:center; color:#0078ff; font-weight:800; font-size:2.4rem; }
  .subhead { text-align:center; color:#5b6b7d; font-size:1.1rem; margin-top:-10px; margin-bottom:24px; }
  .boxed { background:#fff; border:1px solid #e6ebf2; border-radius:14px; padding:16px; }
</style>
""", unsafe_allow_html=True)
st.markdown("<h1>Helios AI (LIDA Integration)</h1>", unsafe_allow_html=True)
st.markdown("<p class='subhead'>Upload → Summarize → Goals → Visualize → Explain → Evaluate → Repair</p>", unsafe_allow_html=True)

# ================= Sidebar =================
with st.sidebar:
    st.subheader("Visualization engine")
    LIBRARY     = st.selectbox("Chart library", ["matplotlib", "seaborn"], index=0)
    TEXT_TEMP   = st.slider("LLM Temperature (text/goals)", 0.0, 1.0, 0.2, 0.05)
    VIZ_TEMP    = st.slider("LLM Temperature (visualize)", 0.0, 1.0, 0.2, 0.05)
    REC_TEMP    = st.slider("LLM Temperature (recommend)", 0.0, 1.0, 0.2, 0.05)
    REC_COUNT   = st.slider("Number of recommendations", 1, 5, 3)

# ================= Helpers =================
def read_any(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx"):
        try:
            import openpyxl  # noqa: F401
        except Exception:
            st.error("Missing dependency 'openpyxl' for .xlsx. Install: pip install openpyxl")
            raise
        return pd.read_excel(file)
    if name.endswith(".json"):
        return pd.read_json(file)
    raise ValueError("Unsupported file format. Use .csv, .xlsx, .json")

def label_goal(g) -> str:
    if isinstance(g, str):
        return g
    if isinstance(g, dict) and "question" in g:
        return g["question"]
    return getattr(g, "question", str(g))

def _looks_png(b: bytes) -> bool:  return len(b) > 8 and b[:8] == b"\x89PNG\r\n\x1a\n"
def _looks_jpeg(b: bytes) -> bool: return len(b) > 3 and b[:3] == b"\xFF\xD8\xFF"
def _looks_svg(b: bytes) -> bool:  return b"<svg" in b[:256].lower()

def _write_temp(ext: str, data: bytes) -> str:
    f = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    f.write(data); f.close()
    return f.name

def _embed_svg(svg_bytes: bytes):
    b64 = base64.b64encode(svg_bytes).decode("ascii")
    st.markdown(f"<img src='data:image/svg+xml;base64,{b64}' style='max-width:100%;'/>", unsafe_allow_html=True)

def run_code_to_png(code: str, df: pd.DataFrame) -> Optional[str]:
    try:
        import seaborn as sns  # noqa: F401
    except Exception:
        sns = None
    ns = {"pd": pd, "np": np, "plt": plt, "data": df, "df": df, "dataset": df, "sns": sns}
    out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    out_path = out.name; out.close()
    try:
        exec(code, ns, ns)
        fig = plt.gcf()
        if fig: fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close("all")
        return out_path if Path(out_path).exists() else None
    except Exception as e:
        plt.close("all")
        st.error(f"Failed to execute chart code: {e}")
        return None

def extract_artifact_and_code(obj):
    if obj is None: return None, None
    if isinstance(obj, dict):
        art  = obj.get("path") or obj.get("image") or obj.get("artifact") or obj.get("img_bytes")
        code = obj.get("code")
        return art, code
    art  = getattr(obj, "path", None) or getattr(obj, "image", None) or getattr(obj, "artifact", None) or getattr(obj, "img_bytes", None)
    code = getattr(obj, "code", None)
    return art, code

def show_chart_artifact(chart_obj, code: Optional[str], df: pd.DataFrame) -> Optional[str]:
    artifact, code_from_obj = extract_artifact_and_code(chart_obj)
    code = code or code_from_obj

    # path-like
    if isinstance(artifact, (str, Path)):
        apath = str(artifact)
        if apath.lower().endswith(".svg"):
            with open(apath, "rb") as f: _embed_svg(f.read())
            return None
        st.image(apath, use_container_width=True)
        return apath

    # bytes-like
    if isinstance(artifact, (bytes, bytearray, io.BytesIO)):
        b = artifact if isinstance(artifact, (bytes, bytearray)) else artifact.getvalue()
        if _looks_png(b) or _looks_jpeg(b):
            ext = ".png" if _looks_png(b) else ".jpg"
            path = _write_temp(ext, b); st.image(path, use_container_width=True); return path
        if _looks_svg(b):
            _embed_svg(b); return None
        if code:
            path = run_code_to_png(code, df)
            if path: st.image(path, use_container_width=True); return path
        st.warning("Got non-image bytes and no usable code to render."); return None

    # no artifact → try code
    if code:
        path = run_code_to_png(code, df)
        if path: st.image(path, use_container_width=True); return path

    st.warning("No image path returned (but code may be available).")
    if code: st.code(code, language="python")
    return None

# ---------- OpenAI helpers for marketing rewrites ----------
def marketing_from_explanation(expl_obj: object) -> str:
    payload = json.dumps(expl_obj, default=str)
    key = "expl_" + payload[:2000]
    cache = st.session_state.get("mk_cache")
    if cache and cache.get("key") == key:
        return cache["text"]
    system = (
        "You are a senior marketing analyst. Rewrite the provided chart explanation "
        "into a concise, non-technical summary (5–8 bullet points) focused on "
        "trends, drivers, segments, outliers, and actionable takeaways."
    )
    resp = oa_client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":payload}],
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    st.session_state["mk_cache"] = {"key": key, "text": text}
    return text

def marketing_from_summary(summary_obj: dict) -> str:
    """
    NEW: Rewrite the LIDA summary into a marketing-analyst overview (bullets),
    with caching keyed by the JSON payload.
    """
    payload = json.dumps(summary_obj, default=str)
    key = "sum_" + payload[:2000]
    cache = st.session_state.get("mk_cache_summary")
    if cache and cache.get("key") == key:
        return cache["text"]
    system = (
        "You are a senior marketing analyst. Translate the dataset schema and field "
        "properties into a concise overview (5–8 bullet points) for a marketing team. "
        "Focus on what matters to marketers: customer/engagement metrics, time ranges, "
        "segments/categories, notable ranges/outliers, and data quality caveats. "
        "Avoid jargon and do not invent facts."
    )
    resp = oa_client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":payload}],
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    st.session_state["mk_cache_summary"] = {"key": key, "text": text}
    return text

# ================= Session =================
st.session_state.setdefault("summary", None)
st.session_state.setdefault("goals", [])
st.session_state.setdefault("goal_obj", None)

st.session_state.setdefault("chart_code", None)
st.session_state.setdefault("chart_png", None)

st.session_state.setdefault("explanation_obj", None)
st.session_state.setdefault("explanation_view", "LIDA JSON")

st.session_state.setdefault("eval_ready", False)
st.session_state.setdefault("eval_df", None)

st.session_state.setdefault("repair_text", "")
st.session_state.setdefault("recommendations", None)

# ================= Upload / flow =================
st.markdown("### 📤 Upload your data")
uploaded = st.file_uploader("Drop or browse a CSV, Excel, or JSON file", type=["csv", "xlsx", "json"])
st.caption("Supports .csv, .xlsx, .json")

if uploaded:
    try:
        df = read_any(uploaded)
        st.success("✅ File loaded successfully!")

        # Stats & preview
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns):,}")
        c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
        st.markdown("#### Preview (first 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

        # --------- Dataset Summary (with toggle) ---------
        st.markdown("### 🧾 Dataset Summary (LIDA)")
        with st.spinner("Summarizing dataset with LIDA…"):
            tcfg_sum = TextGenerationConfig(n=1, temperature=TEXT_TEMP, model=DEFAULT_MODEL, use_cache=True)
            summary = lida.summarize(df, summary_method="llm", textgen_config=tcfg_sum)
            st.session_state["summary"] = summary

        # Toggle lives directly under the header
        sum_view = st.radio(
            "Show:",
            ["LIDA JSON", "Marketing overview (OpenAI)"],
            horizontal=True,
            index=0,
            key="summary_view_radio"
        )
        st.markdown("<div class='boxed'>", unsafe_allow_html=True)
        if sum_view == "LIDA JSON":
            st.json(summary)
        else:
            try:
                mk_sum = marketing_from_summary(summary)
                st.markdown(mk_sum)
            except Exception as e:
                st.error(f"Marketing rewrite failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Goals
        st.markdown("### 🎯 LIDA Goals")
        with st.spinner("Generating analysis goals…"):
            tcfg_goals = TextGenerationConfig(n=1, temperature=TEXT_TEMP, model=DEFAULT_MODEL, use_cache=True)
            goals = lida.goals(summary, n=10, textgen_config=tcfg_goals) or []
            st.session_state["goals"] = goals

        if not goals:
            st.info("No goals returned. Type a custom question:")
            goal_obj = st.text_input("Your goal/question")
        else:
            labels = [label_goal(g) for g in goals]
            chosen = st.selectbox("Select a goal:", labels, index=0)
            goal_obj = goals[labels.index(chosen)]
        st.session_state["goal_obj"] = goal_obj

        # Visualize
        if st.button("Generate visualization", type="primary"):
            with st.spinner("Generating visualization with LIDA…"):
                tcfg_vis = TextGenerationConfig(n=1, temperature=VIZ_TEMP, model=DEFAULT_MODEL, use_cache=True)
                try:
                    charts = lida.visualize(summary=summary, goal=goal_obj, library=LIBRARY,
                                            textgen_config=tcfg_vis, execute_code=True)
                except TypeError:
                    charts = lida.visualize(summary=summary, goal=goal_obj, library=LIBRARY,
                                            textgen_config=tcfg_vis)

            chart = charts[0] if isinstance(charts, list) else charts
            left, right = st.columns([0.62, 0.38])
            with left:
                path = show_chart_artifact(chart, getattr(chart, "code", None), df)

            st.session_state["chart_code"] = getattr(chart, "code", None)
            st.session_state["chart_png"]  = path
            st.session_state["eval_ready"] = False
            st.session_state["eval_df"] = None
            st.session_state["recommendations"] = None

            # Explanation (stored & shown on right)
            lida_expl = None
            try:
                tcfg_exp = TextGenerationConfig(n=1, temperature=TEXT_TEMP, model=DEFAULT_MODEL, use_cache=True)
                lida_expl = lida.explain(code=st.session_state["chart_code"], library=LIBRARY, textgen_config=tcfg_exp)
            except Exception as e:
                st.error(f"Explain failed: {e}")
            st.session_state["explanation_obj"] = lida_expl
            st.session_state["explanation_view"] = "LIDA JSON"

            with right:
                st.markdown("#### 🧠 Explanation")
                view = st.radio(
                    "View",
                    ["LIDA JSON", "Marketing overview (OpenAI)"],
                    horizontal=True,
                    index=0,
                    key="explanation_view_radio"
                )
                st.session_state["explanation_view"] = view
                st.markdown("<div class='boxed'>", unsafe_allow_html=True)
                if view == "LIDA JSON":
                    st.json(lida_expl)
                else:
                    try:
                        mk_text = marketing_from_explanation(lida_expl)
                        st.markdown(mk_text)
                    except Exception as e:
                        st.error(f"Marketing overview failed: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

        # Keep chart+explanation visible after reruns
        if st.session_state.get("chart_code") and st.session_state.get("chart_png") is not None and st.session_state.get("explanation_obj") is not None:
            left, right = st.columns([0.62, 0.38])
            with left:
                st.image(st.session_state["chart_png"], use_container_width=True)
            with right:
                st.markdown("#### 🧠 Explanation")
                view = st.radio(
                    "View",
                    ["LIDA JSON", "Marketing overview (OpenAI)"],
                    horizontal=True,
                    index=0 if st.session_state.get("explanation_view") == "LIDA JSON" else 1,
                    key="explanation_view_radio_persist"
                )
                st.session_state["explanation_view"] = view
                st.markdown("<div class='boxed'>", unsafe_allow_html=True)
                if view == "LIDA JSON":
                    st.json(st.session_state["explanation_obj"])
                else:
                    try:
                        mk_text = marketing_from_explanation(st.session_state["explanation_obj"])
                        st.markdown(mk_text)
                    except Exception as e:
                        st.error(f"Marketing overview failed: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

        # Evaluate
        if st.session_state.get("chart_code"):
            st.markdown("---")
            if st.button("Evaluate visualization", type="primary", key="eval_btn"):
                try:
                    tcfg_eval = TextGenerationConfig(n=1, temperature=TEXT_TEMP, model=DEFAULT_MODEL, use_cache=True)
                    ev = lida.evaluate(code=st.session_state["chart_code"],
                                       goal=st.session_state["goal_obj"],
                                       library=LIBRARY,
                                       textgen_config=tcfg_eval)[0]
                    df_eval = pd.DataFrame(
                        [{"Dimension": e["dimension"], "Score (1–10)": e["score"], "Rationale": e["rationale"]}
                         for e in ev]
                    )
                    st.session_state["eval_df"]   = df_eval
                    st.session_state["eval_ready"] = True
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

        # Repair & Recommendations
        if st.session_state.get("eval_ready"):
            if st.session_state.get("eval_df") is not None:
                st.dataframe(st.session_state["eval_df"], use_container_width=True)

            st.markdown("### 🔧 Repair query & recommended charts")
            st.caption("Use the evaluation to refine the goal or try auto-recommended charts.")
            st.session_state["repair_text"] = st.text_input(
                "Refine your goal/instructions (e.g., 'Use a bar chart by month and highlight outliers').",
                value=st.session_state.get("repair_text", ""),
                key="repair_text_input"
            )

            c_recs, c_edit = st.columns(2)
            with c_recs:
                if st.button("Get visualization recommendations", key="get_recs"):
                    try:
                        with st.spinner("Generating recommended charts…"):
                            tcfg_rec = TextGenerationConfig(
                                n=REC_COUNT, temperature=REC_TEMP, model=DEFAULT_MODEL, use_cache=True
                            )
                            recs = lida.recommend(
                                code=st.session_state["chart_code"],
                                summary=st.session_state["summary"],
                                n=REC_COUNT,
                                textgen_config=tcfg_rec,
                            )
                            st.session_state["recommendations"] = recs or []
                    except Exception as e:
                        st.error(f"Recommendation failed: {e}")

            with c_edit:
                if st.button("Apply repair instruction", key="apply_repair"):
                    try:
                        with st.spinner("Editing current chart…"):
                            tcfg_edit = TextGenerationConfig(n=1, temperature=TEXT_TEMP, model=DEFAULT_MODEL, use_cache=True)
                            edited = lida.edit(
                                code=st.session_state["chart_code"],
                                summary=st.session_state["summary"],
                                instructions=[st.session_state["repair_text"]] if st.session_state["repair_text"] else [],
                                library=LIBRARY,
                                textgen_config=tcfg_edit,
                            )
                            edited_chart = edited[0] if isinstance(edited, list) else edited
                            path = show_chart_artifact(edited_chart, getattr(edited_chart, "code", None), df)
                            st.session_state["chart_code"] = getattr(edited_chart, "code", None)
                            st.session_state["chart_png"]  = path
                            st.session_state["recommendations"] = None

                            # refresh explanation for edited chart
                            try:
                                tcfg_exp = TextGenerationConfig(n=1, temperature=TEXT_TEMP, model=DEFAULT_MODEL, use_cache=True)
                                lida_expl = lida.explain(code=st.session_state["chart_code"], library=LIBRARY, textgen_config=tcfg_exp)
                                st.session_state["explanation_obj"] = lida_expl
                                st.session_state["mk_cache"] = None
                            except Exception as e:
                                st.error(f"Explain failed: {e}")
                    except Exception as e:
                        st.error(f"Edit failed: {e}")

            recs = st.session_state.get("recommendations") or []
            if recs:
                st.markdown("#### 📌 Recommended charts")
                for idx, rec_chart in enumerate(recs, start=1):
                    st.markdown(f"**Recommendation {idx}**")
                    preview_path = show_chart_artifact(rec_chart, getattr(rec_chart, "code", None), df)
                    c1, c2 = st.columns([0.25, 0.75])
                    with c1:
                        if st.button(f"Use recommendation {idx}", key=f"use_rec_{idx}"):
                            st.session_state["chart_code"] = getattr(rec_chart, "code", None)
                            st.session_state["chart_png"]  = preview_path
                            st.session_state["recommendations"] = None
                            # refresh explanation
                            try:
                                tcfg_exp = TextGenerationConfig(n=1, temperature=TEXT_TEMP, model=DEFAULT_MODEL, use_cache=True)
                                lida_expl = lida.explain(code=st.session_state["chart_code"], library=LIBRARY, textgen_config=tcfg_exp)
                                st.session_state["explanation_obj"] = lida_expl
                                st.session_state["mk_cache"] = None
                            except Exception as e:
                                st.error(f"Explain failed: {e}")
                            st.success(f"Applied recommendation {idx}.")
                    with c2:
                        if getattr(rec_chart, "code", None):
                            with st.expander("Show code"):
                                st.code(getattr(rec_chart, "code"), language="python")

    except Exception as e:
        st.error(f"❌ Error: {e}")




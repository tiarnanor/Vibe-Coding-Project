# marketing insights generator
#make sure to run this in terminal: pip install streamlit pandas numpy openai plotly pyarrow
# run pip install "openai>=1.2.0" --upgrade

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

# (Optional) Plotly is here if/when you swap back to live charts
import plotly.express as px  # noqa: F401

# =========================
# Page / Branding
# =========================
st.set_page_config(page_title="Helios AI", page_icon="📤", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f5f7fb; color: #1b1f23; }
    h1 { text-align: center; color: #0078ff; font-weight: 800; font-size: 2.4rem; }
    .subhead { text-align: center; color: #5b6b7d; font-size: 1.1rem; margin-top: -10px; margin-bottom: 24px; }

    [data-testid="stFileUploader"] section { padding: 0 !important; border: none !important; background: transparent !important; }
    [data-testid="stFileUploaderDropzone"] {
        border: 3px dashed #0078ff !important; border-radius: 16px !important; background: #ffffff !important;
        padding: 60px !important; box-shadow: 0 4px 20px rgba(0, 120, 255, 0.1); transition: all 0.3s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #00aaff !important; background: #f0faff !important; box-shadow: 0 6px 25px rgba(0, 120, 255, 0.2);
    }
    [data-testid="stFileUploaderDropzone"] div { text-align: center !important; color: #2d3e50; font-weight: 600; }
    [data-testid="stFileUploaderDropzone"] svg { display: block; margin: 0 auto 12px auto; width: 64px; height: 64px; color: #0078ff; }
    .upload-sub { color:#6c7b8a; margin-top: 6px; text-align:center; font-size: 0.95rem; }

    /* Compact buttons */
    .small-btn .stButton>button { padding: 0.3rem 0.6rem; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Helios AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subhead'>Part 1: Upload data & AI overview · Part 2: (Testing) render the static chart image</p>", unsafe_allow_html=True)

# Static test image (always shown for Part 2 in this build)
IMG_PATH = Path(__file__).parent / "Chart.png"

# =========================
# OpenAI client (for AI overview in Part 1)
# =========================
OPENAI_READY = False
_client = None

try:
    from openai import OpenAI  # pip install "openai>=1.2.0"

    # Get the API key from Streamlit secrets (on Streamlit Cloud)
    api_key = st.secrets.get("OPENAI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "No OPENAI_API_KEY found in Streamlit secrets. "
            "Add it in the app's Advanced settings → Secrets."
        )

    _client = OpenAI(api_key=api_key)
    OPENAI_READY = True

except Exception as e:
    st.warning(f"⚠️ OpenAI not initialized: {e}")
    OPENAI_READY = False


# =========================
# Sidebar to use ChatGPT
# =========================
with st.sidebar:
    st.subheader("Options")
    use_llm_overview = st.toggle("Use ChatGPT for overview (Part 1)", value=True if OPENAI_READY else False)
    model_name = st.text_input("Model", "gpt-4o-mini")
    st.caption("Part 2 (chart) is in testing mode and always shows Chart.png.")

# =========================
# Session state for the two parts + upload + evaluation
# =========================
st.session_state.setdefault("data_file", None)
st.session_state.setdefault("overview_text", None)
st.session_state.setdefault("viz_image_path", None)
st.session_state.setdefault("viz_code", None)
st.session_state.setdefault("viz_expl", None)
st.session_state.setdefault("eval_scores", None)
st.session_state.setdefault("eval_overall", None)

def clear_part1():
    st.session_state["overview_text"] = None

def clear_part2():
    st.session_state["viz_image_path"] = None
    st.session_state["viz_code"] = None
    st.session_state["viz_expl"] = None
    st.session_state["eval_scores"] = None
    st.session_state["eval_overall"] = None

# =========================
# Upload
# =========================
st.markdown("### Upload your data")
uploaded = st.file_uploader(
    "Drop your data file here",
    type=["csv", "tsv", "xlsx", "xlsm", "xls", "json", "parquet"],
    accept_multiple_files=False
)
st.markdown("<p class='upload-sub'>or click to browse</p>", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def read_any(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv") or name.endswith(".tsv"):
        sep = "\t" if name.endswith(".tsv") else ","
        return pd.read_csv(file, sep=sep)
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(file)
    if name.endswith(".json"):
        return pd.read_json(file)
    if name.endswith(".parquet"):
        return pd.read_parquet(file)
    file.seek(0)
    return pd.read_csv(file)

def detect_types(df: pd.DataFrame):
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    dts  = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if not dts:
        for c in df.columns:
            if df[c].dtype == object:
                try:
                    parsed = pd.to_datetime(df[c], errors="raise")
                    if (~pd.isna(parsed)).mean() >= 0.7:
                        dts.append(c)
                except Exception:
                    pass
    cats = [c for c in df.columns if c not in nums + dts]
    return nums, dts, cats

def recommended_prompts(df: pd.DataFrame):
    nums, dts, cats = detect_types(df)
    recs = []
    if nums: recs.append(f"Show distribution of {nums[0]}")
    if dts and nums: recs.append(f"Trend of {nums[0]} over time by {dts[0]}")
    if cats and nums: recs.append(f"Compare {nums[0]} by {cats[0]}")
    if len(nums) >= 2: recs.append(f"Relationship between {nums[0]} and {nums[1]}")
    for g in ["Top categories by count", "Monthly trend of main metric", "Outliers in numerics", "Compare two numeric fields"]:
        if len(recs) >= 4: break
        if g not in recs: recs.append(g)
    return recs[:4]

def schema_payload(df: pd.DataFrame, max_cols=30, max_rows=8):
    schema = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns[:max_cols]]
    sample = df.head(max_rows).to_dict(orient="records")
    return {"schema": schema, "sample": sample}

def ai_overview(df: pd.DataFrame, model: str) -> str:
    if not OPENAI_READY:
        raise RuntimeError("OpenAI key not configured.")
    system = (
        "You are a senior marketing analyst. Based on the dataset schema and sample rows, "
    "provide a concise insight summary (5–8 bullet points max) written for a marketing team. "
    "Focus on what matters to marketers: customers, segments, performance trends, outliers, "
    "growth opportunities, campaign relevance, and data quality caveats. "
    "Avoid technical jargon, avoid guessing, and do not invent insights that are not visible in the sample. "
    "Write in a confident, insight-driven business tone — not a data science tone."
    )
    user = json.dumps(schema_payload(df))
    resp = _client.responses.create(
        model=model,
        input=[{"role": "system", "content": system},
               {"role": "user", "content": user}]
    )
    return resp.output_text.strip()

def evaluate_visualisation() -> dict:
    rubric = {
        "Clarity & Readability": {"score": 4, "why": "Axes and layout are easy to read at a glance."},
        "Relevance to Prompt": {"score": 4, "why": "The visual aligns with the stated question or comparison."},
        "Accuracy & Integrity": {"score": 5, "why": "No mismatch between labels and data; scales look reasonable."},
        "Labeling & Legends": {"score": 3, "why": "Labels/legend exist but could be more descriptive."},
        "Visual Design Choices": {"score": 4, "why": "Chart type and ordering support comparison without clutter."},
        "Actionability / Insight": {"score": 3, "why": "Findings are interpretable but key takeaways could be clearer."},
    }
    return rubric

# =========================
# Main Logic
# =========================
if uploaded is not None:
    try:
        df = read_any(uploaded)
        st.success("✅ File loaded successfully!")

        rows, cols = df.shape
        missing_total = int(df.isna().sum().sum())

        # --- Summary ---
        st.markdown("### Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{rows:,}")
        c2.metric("Columns", f"{cols:,}")
        c3.metric("Missing values", missing_total)

        st.markdown("#### Preview (first 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

        # --- AI Overview ---
        hdr1 = st.columns([0.88, 0.12])
        with hdr1[0]:
            st.markdown("### AI Overview")
        with hdr1[1]:
            with st.container():
                st.markdown('<div class="small-btn">', unsafe_allow_html=True)
                if st.button("🔁 Redo Summary", key="clear_part1"):
                    clear_part1()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        if use_llm_overview and OPENAI_READY and st.session_state["overview_text"] is None:
            try:
                with st.spinner("💬 Asking ChatGPT for a quick overview..."):
                    st.session_state["overview_text"] = ai_overview(df, model_name)
            except Exception as e:
                st.error(f"⚠️ Could not generate AI overview: {e}")

        if st.session_state["overview_text"]:
            st.success("AI summary:")
            st.markdown(st.session_state["overview_text"])
        else:
            st.info("Enable **Use ChatGPT for overview (Part 1)** in the sidebar to generate a natural-language summary.")

        # --- Part 2: Visualisation & Evaluation ---
        st.markdown("---")
        st.markdown("### Ask for an insight or comparison")
        left, right = st.columns([0.62, 0.38])
        with left:
            prompt = st.text_input("Describe the comparison you want (ignored in testing mode).",
                                   placeholder="e.g., Compare total sales by region")
        with right:
            st.caption("Quick suggestions")
            recs = recommended_prompts(df)
            bcols = st.columns(4)
            chosen = None
            for i, r in enumerate(recs):
                if bcols[i].button(r, use_container_width=True, key=f"suggest_{i}"):
                    chosen = r

        final_prompt = chosen or prompt
        go = st.button("Generate visualisation", type="primary")

        if go:
            if not IMG_PATH.exists():
                st.error(f"Chart.png not found at: {IMG_PATH}")
            else:
                st.session_state["viz_image_path"] = str(IMG_PATH)
                st.session_state["viz_code"] = (
                    "# Testing mode: render a static image instead of generating a chart\n"
                    "from pathlib import Path\n"
                    "import streamlit as st\n"
                    "IMG_PATH = Path(__file__).parent / 'Chart.png'\n"
                    "st.image(str(IMG_PATH), use_container_width=True)\n"
                )
                st.session_state["viz_expl"] = "This is a static test image (**Chart.png**) shown regardless of the prompt."
                st.session_state["eval_scores"] = None
                st.session_state["eval_overall"] = None

        # --- Result & Evaluation ---
        if st.session_state.get("viz_image_path"):
            hdr2 = st.columns([0.88, 0.12])
            with hdr2[0]:
                st.markdown("### Result")
            with hdr2[1]:
                with st.container():
                    st.markdown('<div class="small-btn">', unsafe_allow_html=True)
                    if st.button("✖️ Clear", key="clear_part2"):
                        clear_part2()
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            c1r, c2r = st.columns([0.62, 0.38])
            with c1r:
                st.image(st.session_state["viz_image_path"], use_container_width=True, caption="(Test) Static chart preview")
            with c2r:
                show_code = st.toggle("Show code (toggle for explanation)", value=True)
                if show_code:
                    st.subheader("Code used")
                    st.code(st.session_state["viz_code"], language="python")
                else:
                    st.subheader("Explanation")
                    st.write(st.session_state["viz_expl"])

            st.divider()

            if st.button("Evaluate Visualisation", type="primary"):
                scores = evaluate_visualisation()
                st.session_state["eval_scores"] = scores
                vals = [d["score"] for d in scores.values()]
                st.session_state["eval_overall"] = round(sum(vals) / len(vals), 2)

            if st.session_state.get("eval_scores"):
                st.markdown("#### Evaluation Scorecard (for retraining)")
                c_top = st.columns(3)
                with c_top[0]:
                    st.metric("Overall score (1–5)", st.session_state["eval_overall"])
                with c_top[1]:
                    st.caption("Scored on 6 parameters")
                with c_top[2]:
                    st.caption("Use to supervise & retrain")

                import pandas as _pd
                eval_table = _pd.DataFrame(
                    [{"Parameter": k, "Score (1–5)": v["score"], "Rationale": v["why"]}
                     for k, v in st.session_state["eval_scores"].items()]
                )
                st.dataframe(eval_table, use_container_width=True, hide_index=True)

                st.divider()

                st.markdown("#### Rewrite query to improve the visualisation")
                rcol1, rcol2 = st.columns([0.75, 0.25])
                with rcol1:
                    repair_text = st.text_input(
                        "Suggest a better query (e.g., 'Show median revenue by region, top 10 only').",
                        placeholder="Rewrite the query based on the evaluation..."
                    )
                with rcol2:
                    if st.button("Generate improved visualisation"):
                        if not IMG_PATH.exists():
                            st.error(f"Chart.png not found at: {IMG_PATH}")
                        else:
                            st.session_state["viz_image_path"] = str(IMG_PATH)
                            st.session_state["viz_code"] = (
                                "# Testing mode (repair): still render static image\n"
                                "from pathlib import Path\n"
                                "import streamlit as st\n"
                                "IMG_PATH = Path(__file__).parent / 'Chart.png'\n"
                                "st.image(str(IMG_PATH), use_container_width=True)\n"
                            )
                            st.session_state["viz_expl"] = (
                                "Improved visualisation requested. In testing mode this still shows the static image."
                            )
                            st.session_state["eval_scores"] = None
                            st.session_state["eval_overall"] = None
                            st.rerun()

    except Exception as e:
        st.error(f"❌ Could not read the file: {e}")

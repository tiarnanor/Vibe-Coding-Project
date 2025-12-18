# Product Requirements Document (Final)

## 1. Product Overview (Updated)

The goal of this product is to help users explore and understand datasets through AI-assisted visual analysis, without requiring them to write code. Many non-technical users struggle to move from raw data to insights, particularly when deciding what questions to ask, which charts to create, or how to interpret results. This project addresses that gap by combining automated analysis with user control.

The target users are students, analysts, and non-technical stakeholders who want support exploring structured datasets (e.g., CSV or Excel files) in an interactive way. The current prototype is a Streamlit web application that allows users to upload a dataset, generate an AI-driven summary, explore suggested analytical goals, create visualizations, view explanations of those visualizations, evaluate their quality, and refine or repair them using AI recommendations. The system is designed to keep the user “in the loop” at every step rather than acting as a fully autonomous analyst.



## 2. Core Features & Status

* **Dataset upload (CSV / Excel / JSON)** – *Implemented* (non-AI)
* **Automatic dataset summary** – *Implemented* (AI-dependent, via LIDA + LLM)
* **Goal / question generation from data** – *Implemented* (AI-dependent)
* **Visualization generation from goals** – *Implemented* (AI-dependent)
* **Chart explanation** – *Implemented* (AI-dependent)
* **Marketing-language explanation toggle** – *Implemented* (AI-dependent, OpenAI rewrite)
* **Visualization evaluation (scoring + rationale)** – *Implemented* (AI-dependent)
* **Repair instructions for charts** – *Partially implemented* (AI-dependent)
* **Visualization recommendations** – *Partially implemented* (AI-dependent)
* **Persistent user accounts / saved sessions** – *Future*
* **Advanced chart customization controls** – *Future*



## 3. AI Specification (Final)

The AI in this product performs several distinct but related tasks:

* **Inputs:**

  * Uploaded tabular dataset
  * Generated dataset summary
  * User-selected or user-written analytical goals
  * Visualization code and evaluation feedback

* **Outputs:**

  * Dataset summaries (schema-level, not row-level)
  * Suggested analytical questions (goals)
  * Visualization code and rendered charts
  * Natural-language explanations of charts
  * Evaluation scores and rationales
  * Suggested alternative or improved visualizations

AI appears throughout the user flow but is always triggered explicitly by the user (e.g., “Generate visualization,” “Evaluate,” “Recommend”). The core analytical intelligence is provided by **LIDA**, an open-source library for LLM-driven data visualization, integrated directly from GitHub. LIDA uses a large language model to reason about data structure, goals, and chart design.

For language rewriting (e.g., translating technical explanations into marketing-analyst language), the system calls an OpenAI-compatible LLM through a LiteLLM proxy. Guardrails include:

* No autonomous actions without user input,
* No long-term data storage,
* Explicit toggles between raw (JSON) and rewritten explanations,
* Controlled temperature settings to reduce hallucination and variability.



## 4. Technical Architecture (Reality Check)

* **Front-end:** Streamlit (Python-based web UI)
* **Backend logic:** Python running in the same Streamlit app
* **AI orchestration:**

  * LIDA `Manager` for summarize, goals, visualize, explain, evaluate, recommend, and edit
  * OpenAI-compatible LLM accessed via LiteLLM proxy for explanation rewriting
* **Visualization:** Matplotlib (primary), with optional Seaborn support
* **External services:**

  * OpenAI-compatible API (via proxy URL)
  * Streamlit Cloud for deployment

All AI calls are made server-side, and API keys are injected via environment variables or Streamlit secrets.


## 5. Prompting & Iteration Summary

During development, I relied on “vibe coding” tools—primarily ChatGPT—alongside Visual Studio Code. Some of the most important prompts included:

* Asking how to correctly integrate LIDA with Streamlit session state,
* Debugging why visualization artifacts were returned as bytes instead of image paths,
* Designing prompts to rewrite LIDA explanations into marketing-analyst language without inventing insights.

Over time, prompts became more constrained and explicit. Early prompts were broad (“Why isn’t this chart showing?”), while later prompts focused on specific failure modes or design goals. The biggest lesson was that good prompt design requires clarity about inputs, outputs, and the role the AI is meant to play—especially when combining multiple AI systems.



## 6. UX & Limitations

The intended user journey is:

1. Upload a dataset
2. Review the AI-generated dataset summary
3. Select or refine an analytical goal
4. Generate and view a visualization
5. Read an explanation (technical or marketing-focused)
6. Evaluate, repair, or explore recommended alternatives

Known limitations include occasional visualization failures, reliance on correct environment configuration, and limited styling control over charts. Some interactions can feel “janky,” particularly when Streamlit reruns reset parts of the UI.

From a trust perspective, users should not treat AI-generated explanations or evaluations as authoritative. The system is intended for exploratory analysis and learning, not for high-stakes or production decision-making without human review.



## 7. Future Roadmap

If given more time, the next steps would include:

* Improving robustness and consistency of visualization rendering,
* Adding persistent sessions and saved analyses,
* Strengthening evaluation metrics and explanations,
* Expanding recommendation quality and diversity,
* Conducting user testing to better align outputs with real analyst workflows.

---

This PRD reflects the product as it exists at the end of the course and documents both its capabilities and its limitations honestly.

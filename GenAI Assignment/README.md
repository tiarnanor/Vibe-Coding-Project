\# Helios AI – Data Insight Tool



This is a Streamlit-based prototype for the Vibe Coding Project (Phase 2).  

It demonstrates AI-assisted data summarization, automated insight suggestions, and visualization evaluation for marketing analysts.





\## 🚀 How to Run the Project



\### 1. Clone the repository

\### 2. Make sure to run API\_KEY.py with your openai API key before running test.py

\### 3. run this in terminal: pip install streamlit pandas numpy openai plotly pyarrow

\### 4. Also in terminal run pip install "openai>=1.2.0" --upgrade

\### 5. run " streamlit run test.py " in terminal





---



\## 📌 Features



\### Part 1 — AI Overview

\- Upload a dataset (`csv`, `tsv`, `xlsx`, `json`, `parquet`).

\- Automatic schema detection.

\- Optional AI-generated business insight summary (via OpenAI API).

\- Ability to re-generate or clear the summary.



\### Part 2 — Visualisation

\- Suggests likely analysis prompts based on dataset structure.

\- Shows a static placeholder chart (`Chart.png`) for testing mode.

\- Displays the code used to render the visualization.

\- Provides an automated rubric-based evaluation system.

\- Allows rewriting the query and regenerating an improved visualization.



---



\## 🧠 AI Component Explanation



AI is integrated in \*\*Part 1\*\* of the product:



\### What the AI Does

\- Receives dataset schema + sample rows.

\- Generates a 5–8 bullet business-marketing summary.

\- Writes in “insight-driven marketing” style, not technical analysis.



\### Why AI Is Used

\- Mimics a data strategist writing high-level insights for non-technical teams.

\- Automates a slow, manual part of the analytics workflow.

\- Helps early-stage users quickly understand their data.



\### Where It Appears

\- In the sidebar (“Use ChatGPT for overview”)

\- In the “AI Overview” panel after uploading data.



\### Model Used

\- `gpt-4o-mini` by default (configurable in the UI).



---






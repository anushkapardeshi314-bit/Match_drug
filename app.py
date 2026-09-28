
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="DrugMatch AI", page_icon="💊", layout="wide")

st.title("💊 DrugMatch AI")
st.caption("AI-assisted research prototype for candidate–disease matching and drug-repurposing exploration")

st.warning(
    "Research/demo only. This prototype uses a small synthetic dataset and must not be used "
    "for diagnosis, prescribing, dosing, treatment decisions, or real-world drug selection."
)

DATA = pd.DataFrame([
    ["DM-001","Candidate Alpha","kinase inhibitor","inflammation; autoimmune; kinase","JAK1;JAK2",0.82,0.18,0.71],
    ["DM-002","Candidate Beta","enzyme inhibitor","metabolic; inflammation","AMPK;PDE4",0.74,0.24,0.66],
    ["DM-003","Candidate Gamma","receptor modulator","neuro; inflammation","NR1;HTR2A",0.69,0.31,0.58],
    ["DM-004","Candidate Delta","protease inhibitor","viral; inflammation","PRT1;PRT2",0.77,0.21,0.73],
    ["DM-005","Candidate Epsilon","transporter modulator","metabolic; cardiovascular","SLC1;SLC2",0.63,0.36,0.55],
    ["DM-006","Candidate Zeta","kinase inhibitor","oncology; inflammation","EGFR;JAK1",0.86,0.16,0.79],
    ["DM-007","Candidate Eta","enzyme inhibitor","neuro; metabolic","MAO-A;AMPK",0.67,0.29,0.61],
    ["DM-008","Candidate Theta","receptor modulator","cardiovascular; inflammation","ADRB1;HTR2A",0.72,0.26,0.64],
    ["DM-009","Candidate Iota","protease inhibitor","viral; oncology","PRT2;CASP1",0.81,0.20,0.70],
    ["DM-010","Candidate Kappa","enzyme inhibitor","metabolic; oncology","AMPK;TYK2",0.75,0.23,0.68],
], columns=["id","candidate","class","indications","targets","evidence","risk_signal","novelty"])

st.sidebar.header("Research Query")
disease = st.sidebar.selectbox(
    "Choose a research area",
    ["inflammation","autoimmune","metabolic","neuro","oncology","viral","cardiovascular"]
)
target = st.sidebar.text_input("Optional target keyword", placeholder="e.g. JAK1")
min_evidence = st.sidebar.slider("Minimum evidence score", 0.0, 1.0, 0.60, 0.01)

query = disease + " " + target
corpus = DATA["indications"] + " " + DATA["targets"] + " " + DATA["class"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
q = vectorizer.transform([query])
DATA["semantic_match"] = cosine_similarity(q, X).ravel()

# Transparent research score: not a clinical score.
DATA["research_match"] = (
    0.55 * DATA["semantic_match"] +
    0.25 * DATA["evidence"] +
    0.20 * DATA["novelty"]
)

results = DATA[DATA["evidence"] >= min_evidence].sort_values(
    "research_match", ascending=False
).reset_index(drop=True)

st.subheader("🔎 Candidate exploration")
c1, c2, c3 = st.columns(3)
c1.metric("Candidates in demo dataset", len(DATA))
c2.metric("Candidates passing filter", len(results))
c3.metric("Research area", disease.title())

if results.empty:
    st.info("No candidates pass the current evidence filter. Lower the filter to explore the demo dataset.")
else:
    display = results[[
        "id","candidate","class","targets","indications",
        "evidence","novelty","semantic_match","research_match"
    ]].copy()
    for col in ["evidence","novelty","semantic_match","research_match"]:
        display[col] = display[col].round(3)

    st.dataframe(display, use_container_width=True, hide_index=True)

    selected = st.selectbox("Inspect a candidate", results["candidate"].tolist())
    row = results[results["candidate"] == selected].iloc[0]

    st.markdown("### 🧬 Candidate research card")
    a,b,c,d = st.columns(4)
    a.metric("Evidence", f"{row.evidence:.2f}")
    b.metric("Novelty", f"{row.novelty:.2f}")
    c.metric("Semantic match", f"{row.semantic_match:.2f}")
    d.metric("Research match", f"{row.research_match:.2f}")

    st.write("**Candidate class:**", row["class"])
    st.write("**Research indications in demo data:**", row["indications"])
    st.write("**Associated target labels:**", row["targets"])

    st.info(
        "Why it appeared: the prototype compares the research query with candidate indication/target "
        "text using TF-IDF cosine similarity, then combines that similarity with preloaded evidence "
        "and novelty fields. These values are synthetic demonstration data."
    )

st.divider()
st.subheader("📊 Compare candidates")
compare = st.multiselect("Select up to 4 candidates", DATA["candidate"].tolist(), max_selections=4)
if compare:
    comp = DATA[DATA["candidate"].isin(compare)][
        ["candidate","class","evidence","novelty","risk_signal","semantic_match","research_match"]
    ].copy()
    st.bar_chart(comp.set_index("candidate")[["evidence","novelty","semantic_match","research_match"]])

st.divider()
st.subheader("🧪 Suggested hackathon extensions")
st.markdown("""
- Replace the synthetic table with a properly licensed public biomedical dataset.
- Add molecular descriptors/fingerprints with RDKit in a research environment.
- Add target–disease knowledge-graph relationships.
- Add an uncertainty/explanation panel for every model output.
- Add dataset provenance and citation tracking.
- Add a model evaluation page with held-out test data and metrics.
""")

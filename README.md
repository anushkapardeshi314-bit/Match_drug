
# DrugMatch AI — Hackathon Project

## One-line pitch
An explainable AI research dashboard that helps researchers explore relationships between disease areas, biological targets, and candidate compounds using transparent similarity scoring.

## Important scope
This is an educational/hackathon prototype. The included dataset is synthetic. It does **not** provide medical advice, dosing, prescriptions, treatment recommendations, or instructions for making drugs.

## Features
1. Research-area selection
2. Optional target keyword
3. Candidate filtering by evidence score
4. TF-IDF + cosine similarity for semantic matching
5. Transparent research-match score
6. Candidate research cards
7. Side-by-side comparison
8. Extension roadmap for real biomedical datasets

## Tech stack
- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn
- TF-IDF + cosine similarity

## Run locally
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Suggested 3-person team split
- Frontend: Streamlit dashboard + UI
- AI/ML: similarity model + evaluation
- Backend/Data: dataset schema + provenance + API/data integration

## 3-minute demo flow
1. Select "inflammation".
2. Add target keyword "JAK1".
3. Show ranked research candidates.
4. Open a candidate research card.
5. Explain why the candidate appeared.
6. Compare 3 candidates.
7. Show the roadmap for replacing synthetic data with licensed biomedical datasets.

## Future version
Use a properly licensed biomedical dataset such as an open-access source or an institutionally licensed dataset. Maintain source attribution, versioning, and validation. Do not treat model output as evidence of safety or efficacy.

## Suggested judging points
- Problem relevance
- Explainability
- Data provenance
- Reproducibility
- Clear separation between research exploration and clinical decision-making
- Quality of evaluation

# Measuring Hate Speech — Ethical Audit Pipeline

University project for *Data Science and Ethics* (prof. R. Rosati), MA in Philosophy and Artificial Intelligence, Sapienza University of Rome, a.a. 2025/2026.

**Author**: Andrea Morbello

## What this is

End-to-end ethical audit of the Berkeley *Measuring Hate Speech* corpus (Kennedy et al., 2020), covering privacy (pseudo-anonymization, k-anonymity, l-diversity), bias and fairness, baseline modeling, and explainability (SHAP, LIME).

## How to run

```bash
git clone https://github.com/AndreaMorbello/measuring-hate-speech-audit.git
cd measuring-hate-speech-audit
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/audit.ipynb
```

## Pipeline

1. Data loading & EDA
2. Privacy: pseudo-anonymization and k-anonymity on annotator demographics
3. Bias analysis on target identity groups
4. Baseline classifier (TF-IDF + Logistic Regression)
5. Fairness metrics and mitigation
6. Explainability with SHAP and LIME

## Dataset

[Berkeley Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech), CC-BY-4.0.

## License

MIT

# Hybrid AI Recommendation System
### MSc Dissertation — Dublin Business School

> Built entirely from scratch. Raw data → SQL pipeline → ML model → evaluation → Power BI dashboard.

---

## 🎯 The Problem

Single-method recommendation systems fail when user history is sparse (cold start problem).
Most systems use either collaborative filtering OR content-based filtering — not both.

## 💡 The Solution

A **hybrid model** that combines both techniques, switching intelligently based on data availability.
Result: outperformed both individual baselines on precision and recall.

---

## 🏗️ What I Built

```
Raw Data Sources
      │
      ▼
SQL Cleaning & Transformation
      │
      ▼
Feature Engineering (Python · Pandas · NumPy)
      │
      ├──► Collaborative Filtering Model
      │
      ├──► Content-Based Filtering Model
      │
      ▼
Hybrid Fusion Layer (Scikit-learn)
      │
      ▼
Model Evaluation (Precision · Recall · Coverage)
      │
      ▼
Power BI Dashboard (Stakeholder reporting)
```

---

## 🔧 Tech Stack

| Component | Tech |
|-----------|------|
| Language | Python |
| Data processing | Pandas, NumPy |
| ML models | Scikit-learn |
| Data extraction | SQL |
| Visualisation | Power BI, Matplotlib |
| Version control | Git / GitHub |

---

## 📊 Results

- ✅ Hybrid model outperformed collaborative-only and content-only baselines
- ✅ Automated pipeline reduced processing time by ~30%
- ✅ Rigorous output validation at every stage — no hallucinations, no bad data
- ✅ Results presented to non-technical academic stakeholders via dashboard

---

## 💡 Key Learnings

- Designing reliable AI pipelines, not just accurate models
- Output validation is as important as model performance
- How to translate complex AI outputs into clear business narratives

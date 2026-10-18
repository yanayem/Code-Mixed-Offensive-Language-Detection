# Code-Mixed Offensive Language Detection

A comprehensive benchmark and methodology for **Bangla-English-Hindi code-mixed offensive language detection** using NLP baselines, leakage-safe evaluation protocols, and data quality audits.

---

## 📌 Project Overview

Social media content in South Asian contexts frequently exhibits **code-mixing**—alternating between English (Latin script), Bengali, and Hindi (Devanagari script) within a single post or sentence. Detecting offensive content in such code-mixed environments presents unique NLP challenges, including non-standard orthography, multi-script mixing, informal Romanized transliterations, and data leakage risks from user metadata and source augmentation.

This repository provides an end-to-end pipeline for exploratory data analysis (EDA), data quality auditing, multilingual preprocessing, leakage-safe cross-validation, and baseline classification for code-mixed offensive language detection.

---

## 📊 Dataset Overview

The dataset consists of **100,000 code-mixed social media posts** categorized into binary target labels:
- `NOT` : Not Offensive (~66.7%)
- `OFF` : Offensive (~33.3%)

### Split Summary

| Split | Rows | Target Label Ratio | Description |
|---|---:|---|---|
| **Train** | 60,000 | `NOT`: 40,018 (66.7%) · `OFF`: 19,982 (33.3%) | Primary training set with full schema |
| **Dev** | 20,000 | `NOT`: 13,339 (66.7%) · `OFF`: 6,661 (33.3%) | Official dev split for model validation |
| **Test** | 20,000 | `NOT`: 13,340 (66.7%) · `OFF`: 6,660 (33.3%) | Evaluation set for benchmark reporting |
| **Total** | **100,000** | **Class Ratio ~ 2:1 (`NOT` : `OFF`)** | |

### Data Schema
- `id` (*float64*): Source / sample identifier (not unique per row; variants of the same underlying post share IDs).
- `text` (*string*): Multilingual code-mixed text input (Bengali, Devanagari, Latin).
- `user` (*string*): User metadata handle (has strong label association / identity-leakage risk).
- `state` (*string*): Categorical location metadata (CT, DE, FL, NY, VA).
- `time` (*string*): Timestamp string (`HH:MM:SS`).
- `label` (*string*): Binary target class (`OFF` or `NOT`).

---

## ⚠️ Data Quality & Leakage Audit Insights

1. **Missing Metadata Handling**:
   - ~20.67% of rows in train and test splits have missing metadata (`id`, `user`, `state`, `time`).
   - All rows retain valid `text` and `label`. Rows with missing metadata are **retained** because their text content is fully usable.
2. **Source Augmentation & Repeated IDs**:
   - `id` is not unique per row; multiple rows represent variations of the same underlying text item.
   - Naive random row-based splitting causes data leakage between train and validation folds.
3. **User Metadata Identity Leakage**:
   - Several `user` identities in the dataset are 100% associated with a single label (`OFF` or `NOT`).
   - Using `user` as a primary feature risks memorizing handle-level labels rather than learning offensive language semantics.
4. **Multilingual Code-Mixing Distribution**:
   - ~99.98% of posts contain Latin characters.
   - ~93.65% of posts contain Bengali script (`\u0980-\u09FF`).
   - ~94.13% of posts contain Devanagari/Hindi script (`\u0900-\u097F`).
   - ~90.67% contain `@USER` mentions; ~14.55% contain hashtags; ~14.34% contain URL placeholders.

---

## 🧹 Multilingual Preprocessing Strategy

Traditional English NLP preprocessing (e.g. English stopword removal, stemming, dropping non-ASCII characters) corrupts code-mixed Bangla-English-Hindi text. 

**Safe Preprocessing Protocol:**
- **Unicode NFKC Normalization**: Standardizes character representations across scripts.
- **Zero-Width Character Removal**: Strips formatting artifacts (`\u200B-\u200D`, `\uFEFF`).
- **Placeholder Standardization**: Standardizes `@USER` mentions and `URL` tokens.
- **Lowercasing & Whitespace Normalization**: Preserves script integrity while normalizing spacing.

---

## 🛡️ Leakage-Safe Evaluation Protocol

To prevent optimistic metric inflation caused by duplicate texts and repeated source IDs:
- **Connected-Component Grouped Validation**:
  - Rows sharing the same `user`, same `id`, or same normalized text are linked into a single group.
  - Grouped Shuffle Split / Stratified Group K-Fold ensures no source group overlaps across train and validation folds.

---

## 📈 Model Baselines & Benchmarks

| Model / Architecture | Preprocessing / Features | Validation Strategy | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---:|---:|---:|
| Naive Bayes + BoW | Word (1-2 grams) | Grouped 5-Fold CV | 78.20% | 74.50% | -- |
| TF-IDF + LinearSVC | Word (1-2) + Char (3-5) | Random Split *(Unsafe)* | 89.31% | 87.86% | 89.26% |
| **TF-IDF + LinearSVC** | **Word (1-2) + Char (3-5)** | **Leakage-Safe Grouped Split** | **80.26%** | **77.69%** | **80.07%** |
| TF-IDF + LinearSVC | Word (1-2) + Char (3-5) | Full-Train Test Evaluation | 91.81% | 90.72% | 91.78% |

> **Key Finding**: Random splits inflate validation metrics (~89.3% vs ~80.3%) due to source ID and text overlap across splits. Leakage-safe grouped validation provides the true benchmark for research.

---

## 🚀 Repository Structure

```text
.
├── basilineversion.ipynb    # Main notebook containing EDA, audits, preprocessing & baselines
├── train_processed.csv      # Cleaned and processed training dataset (generated)
├── test_processed.csv       # Cleaned and processed test dataset (generated)
├── submission.csv           # Model predictions in submission format (generated)
└── README.md                # Project documentation
```

---

## 🛠️ Usage & Replication

### Installation

Ensure Python 3.8+ and required libraries are installed:

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### Execution

Launch the Jupyter Notebook to run the pipeline:

```bash
jupyter notebook basilineversion.ipynb
```

The notebook automatically performs:
1. Dataset auto-discovery (`train.csv`, `dev.csv`, `test.csv`).
2. Schema & data quality auditing.
3. Multilingual text cleaning and feature extraction.
4. Leakage-safe grouped validation & baseline training.
5. Generation of processed datasets and model predictions.

---

## 🔬 Recommended Future Directions

1. **Multilingual Transformers**: Fine-tune `xlm-roberta-base`, `mBERT`, or `BanglaBERT` using the leakage-safe grouped split protocol.
2. **Metadata Ablations**: Systematically evaluate Text-Only vs. Text + Metadata (`state`, `time`) to isolate metadata contributions without user handle leakage.
3. **Cross-Lingual & Transliteration Analysis**: Analyze performance per script (Bangla vs. Devanagari vs. Romanized text) to target specific offensive language patterns.

---

## 📄 License

This project is licensed under the MIT License.

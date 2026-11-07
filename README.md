# Code-Mixed Offensive Language Detection

### 👥 Contributors / Authors
- **[Arefa Rahman](https://github.com/ArefaRahmanApy)** (ID: 20245103017)
- **[Taspiya Jannat Prottasa](https://github.com/prottasajannat)** (ID: 20245103057)
- **[Md. Iftekher](https://github.com/iftekharifti02)** (ID: 20245103077)
- **[Yacin Arafat](https://github.com/yanayem)** (ID: 20245103160)

### 👩‍🏫 Supervised By
- **Fateha Jannat Ayrin**  
  *Lecturer, Department of Computer Science and Engineering (CSE)*  
  *Bangladesh University of Business and Technology (BUBT)*

---

A comprehensive benchmark and methodology for **Bangla-English-Hindi code-mixed offensive language detection** using NLP baselines, multilingual transformer representations (MuRIL & XLM-RoBERTa), hybrid feature fusion, leakage-safe evaluation protocols, and data quality audits.

---

## 📌 Project Overview

Social media content in South Asian contexts frequently exhibits **code-mixing**—alternating between English (Latin script), Bengali, and Hindi (Devanagari script) within a single post or sentence. Detecting offensive content in such code-mixed environments presents unique NLP challenges, including non-standard orthography, multi-script mixing, informal Romanized transliterations, and data leakage risks from user metadata and source augmentation.

This repository provides an end-to-end pipeline for exploratory data analysis (EDA), data quality auditing, multilingual preprocessing, leakage-safe cross-validation, baseline machine learning models, and advanced transformer-based hybrid models (`Version/code-offensive-v6.ipynb`) for code-mixed offensive language detection.

---

## 📚 Literature Review / Reviewed Papers

The following table summarizes existing benchmark literature, datasets, models, and baseline results reviewed for this research:

| Paper Title | Model(s) Used | Dataset Used | Key Results |
|---|---|---|---|
| **Mixed-Distil-BERT**<br>*(Raihan et al., 2024)* | Mixed-Distil-BERT, Tri-Distil-BERT | 3-Language (Bangla, English, Hindi) Code-Mixed Synthetic Dataset | Offensive Language Detection:<br>**Accuracy: 0.87**, **Weighted F1: 0.87** |
| **TB-OLID**<br>*(Raihan et al., 2023)* | fBERT, HateBERT, BERT, mBERT | TB-OLID (5,000 Transliterated & Code-Mixed Bangla Facebook Comments) | Offensive Language Identification:<br>**F1-Score: 0.72** *(fBERT & HateBERT)* |
| **EmoMix-3L**<br>*(Raihan et al., 2024)* | MuRIL, XLM-R, GPT-3.5 Turbo, BanglishBERT | EmoMix-3L (1,071 3-Language Code-Mixed Test Instances) & 100k Synthetic Training Set | Emotion Classification (Macro-F1):<br>**MuRIL: 0.54** |
| **OffMix-3L**<br>*(Goswami et al., 2023)* | BanglishBERT, mBERT, XLM-R | OffMix-3L (1,001 3-Language Code-Mixed Test Instances) & 100k Synthetic Training Set | Offensive Language Identification:<br>**F1-Score: 0.68** *(BanglishBERT)* |
| **SentMix-3L**<br>*(Raihan et al., 2023)* | GPT-3.5 Turbo, XLM-R, BanglishBERT, mBERT | SentMix-3L (1,007 3-Language Code-Mixed Test Instances) & 100k Synthetic Training Set | Sentiment Analysis:<br>**Weighted F1-Score: 0.62** *(GPT-3.5 Turbo Zero-Shot)* |

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

## 📈 Baseline Models & Benchmarks

| Model / Architecture | Preprocessing / Features | Validation Strategy | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---:|---:|---:|
| Naive Bayes + BoW | Word (1-2 grams) | Grouped 5-Fold CV | 78.20% | 74.50% | -- |
| TF-IDF + LinearSVC | Word (1-2) + Char (3-5) | Random Split *(Unsafe)* | 89.31% | 87.86% | 89.26% |
| **TF-IDF + LinearSVC** | **Word (1-2) + Char (3-5)** | **Leakage-Safe Grouped Split** | **80.26%** | **77.69%** | **80.07%** |
| TF-IDF + LinearSVC | Word (1-2) + Char (3-5) | Full-Train Test Evaluation | 91.81% | 90.72% | 91.78% |

> **Key Finding**: Random splits inflate validation metrics (~89.3% vs ~80.3%) due to source ID and text overlap across splits. Leakage-safe grouped validation provides the true benchmark for research.

---

## 🤖 Transformer & Hybrid Model Experiments (`Version/code-offensive-v6.ipynb`)

The notebook `Version/code-offensive-v6.ipynb` implements a research pipeline evaluating frozen multilingual transformer embeddings, feature-level fusion, and late-fusion probability ensembling under a strict **5-Fold User-Grouped Cross-Validation** protocol on 60,000 training rows.

### Experimental Setup & Tracks
- **Track A (MuRIL + MLP)**: Mean-pooled embeddings from `google/muril-base-cased` (768-dim) + 2-layer MLP classifier.
- **Track B (XLM-R + MLP)**: Mean-pooled embeddings from `FacebookAI/xlm-roberta-base` (768-dim) + 2-layer MLP classifier.
- **Track C (Hybrid MuRIL-XLMR + MLP)**: Concatenated feature vectors (`768 + 768 = 1536` dims) + trainable MLP classifier with LayerNorm, GELU, and Focal Loss ($\gamma = 2.0$).
- **Track D (Late Fusion Ensemble)**: Average of Track A and Track B out-of-fold predicted probabilities.

### Validation & Training Protocol
- **Fold Assignment**: Known users grouped together (`USER::` > `ID::` > `TEXT::`); 0% group overlap across train and validation folds.
- **Preprocessing & Loss**: `StandardScaler` and class-balanced Focal Loss $\alpha$ weights computed independently inside each fold to avoid statistics leakage.
- **Optimization**: AdamW ($\text{lr}=10^{-3}$, weight decay $=10^{-4}$), early stopping based on validation Macro-F1 (patience = 5).

### 🏆 5-Fold Pooled Out-Of-Fold (OOF) Benchmark Results

| Model / Architecture | Feature Representation | Validation Strategy | Accuracy | Macro-F1 | Weighted-F1 | OFF-Precision | OFF-Recall | OFF-F1 | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Late Fusion (Mean Probabilities)** | Probability Average (A + B) | 5-Fold Grouped CV | **79.96%** | **76.88%** | **79.70%** | **71.95%** | 65.27% | **68.45%** | **0.8489** | **0.7532** |
| **Hybrid MuRIL-XLMR + MLP** | Feature Concatenation (1536-d) | 5-Fold Grouped CV | **78.87%** | **75.95%** | **78.75%** | 69.11% | **66.09%** | **67.57%** | **0.8358** | **0.7304** |
| MuRIL + MLP | `google/muril-base-cased` (768-d) | 5-Fold Grouped CV | 77.56% | 74.42% | 77.41% | 67.16% | 63.84% | 65.46% | 0.8163 | 0.7044 |
| XLM-R + MLP | `FacebookAI/xlm-roberta-base` (768-d) | 5-Fold Grouped CV | 76.29% | 73.17% | 76.23% | 64.71% | 63.37% | 64.03% | 0.8087 | 0.6890 |

#### Fold-wise Summary (Mean ± Standard Deviation)

| Model / Architecture | Accuracy | Macro-F1 | OFF-F1 | OFF-Recall |
|---|---:|---:|---:|---:|
| **Hybrid MuRIL-XLMR + MLP** | **78.95% ± 1.83%** | **75.90% ± 1.54%** | **67.44% ± 2.18%** | **66.32% ± 2.29%** |
| MuRIL + MLP | 77.64% ± 1.51% | 74.32% ± 1.17% | 65.21% ± 2.32% | 63.85% ± 2.42% |
| XLM-R + MLP | 76.39% ± 1.67% | 73.10% ± 1.21% | 63.83% ± 2.19% | 63.45% ± 3.16% |

### 💡 Key Findings from Notebook v6
1. **Multilingual Representations**: MuRIL outperforms XLM-RoBERTa individually (+1.25% Macro-F1, +1.27% Accuracy) due to pre-training specifically tailored for Indian languages and transliterated scripts.
2. **Feature Concatenation vs. Single Encoders**: Concatenating frozen embeddings (Hybrid Track C) boosts Macro-F1 by **+1.53%** over MuRIL alone and **+2.78%** over XLM-R alone.
3. **Ensemble Performance**: The Late Fusion probability ensemble achieves the highest overall benchmark score (**76.88% Macro-F1**, **79.96% Accuracy**, **0.8489 ROC-AUC**).
4. **Focal Loss Effectiveness**: Focal Loss ($\gamma = 2.0$) improves minority class (`OFF`) recall across all folds.

---

## 🚀 Repository Structure

```text
.
├── project_baseline.ipynb                    # Project baseline notebook for Assignment Part B (Paper review, baseline models, exhibits E, F, G)
├── basilineversion.ipynb                     # Main notebook containing EDA, audits, preprocessing & ML baselines
├── Version/
│   ├── code-offensive-v6.ipynb               # MuRIL + XLM-RoBERTa Hybrid & Ensembles with 5-Fold User-Grouped CV
│   └── code-offensive-v7.ipynb               # End-to-End Joint Transformer Fine-Tuning & Gated Fusion (v7)
├── train_processed.csv                       # Cleaned and processed training dataset (generated)
├── test_processed.csv                        # Cleaned and processed test dataset (generated)
├── submission.csv                            # Model predictions in submission format (generated)
├── hybrid_5fold_fold_metrics.csv             # 5-fold metric breakdown for transformer models (generated)
├── hybrid_5fold_oof_summary.csv              # Pooled OOF evaluation summary (generated)
├── hybrid_5fold_oof_predictions.csv          # Out-of-fold predictions and probabilities (generated)
├── hybrid_5fold_experiment_manifest.json     # Experiment metadata manifest (generated)
├── NEXT_STEPS.md                             # Research roadmap & conference publication plan
└── README.md                                 # Project documentation
```

---

## 🛠️ Usage & Replication

### Installation

Ensure Python 3.8+ and required libraries are installed:

```bash
pip install pandas numpy scikit-learn matplotlib torch transformers jupyter
```

### Execution

#### 1. Project Baseline Pipeline (Assignment Part B)
Launch the baseline notebook containing Exhibit E, Exhibit F, Model Card, and baseline experiments:

```bash
jupyter notebook project_baseline.ipynb
```

#### 2. EDA & Detailed ML Baseline Pipeline
Launch the baseline notebook for EDA, data quality audits, preprocessing, and TF-IDF / LinearSVC models:

```bash
jupyter notebook basilineversion.ipynb
```

#### 3. Transformer & Hybrid Pipeline (v6)
Launch the Version 6 notebook for MuRIL + XLM-RoBERTa feature extraction, 5-fold user-grouped cross-validation, hybrid MLP training, and probability ensembling:

```bash
jupyter notebook Version/code-offensive-v6.ipynb
```

The notebook automatically performs:
1. Dataset auto-discovery (`train.csv`, `dev.csv`, `test.csv`).
2. Data quality & cross-split overlap auditing.
3. User-grouped 5-fold stratified key construction.
4. Frozen embedding extraction (`google/muril-base-cased` & `FacebookAI/xlm-roberta-base`) with local caching.
5. 5-fold MLP training with Focal Loss and fold-fitted scaling.
6. Probability averaging ensemble and confusion matrix / ROC / PR analysis.
7. Exporting OOF predictions and evaluation summary CSVs.

#### 3. End-to-End Fine-Tuning & Gated Fusion Pipeline (v7)
Launch the Version 7 notebook for unfreezing transformer backbones, end-to-end joint fine-tuning with differential learning rates, gated attention fusion, and test set inference:

```bash
jupyter notebook Version/code-offensive-v7.ipynb
```

---

## 🔬 Version 7 Roadmap & Future Directions

1. **End-to-End Joint Fine-Tuning**: Transitioning from frozen embedding extraction (v6) to end-to-end fine-tuning of `google/muril-base-cased` and `FacebookAI/xlm-roberta-base` with layer-wise differential learning rates ($\text{LR}_{\text{backbone}} = 2 \times 10^{-5}$).
2. **Gated Attention Fusion**: Implementing a Gated Attention Fusion Module to dynamically weight MuRIL vs. XLM-R token representations per sample instead of naive concatenation.
3. **Per-Script Ablation**: Evaluating offensive language classification accuracy separately on Bangla script (`\u0980-\u09FF`), Devanagari script (`\u0900-\u097F`), and Romanized/Banglish text segments.
4. **Official Test Set Evaluation**: Generating out-of-fold ensemble predictions on the official 20,000 `test.csv` instances for benchmark reporting.

---

## 📄 License

This project is licensed under the MIT License.

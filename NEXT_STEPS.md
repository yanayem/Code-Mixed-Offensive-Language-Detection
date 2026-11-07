# 🗺️ Research Roadmap & Conference Publication Plan (`NEXT_STEPS.md`)

This document outlines the evaluation of current model results (`Version/code-offensive-v6.ipynb`), the paper drafting strategy for conference submission, and the concrete technical implementation plan for **Version 7** (`Version/code-offensive-v7.ipynb`).

---

## 📊 1. Performance Evaluation: v6 Model vs. Benchmark Literature

### 📈 Comparative Overview

| Benchmark / Paper | Model(s) Used | Dataset & Setup | Key Paper Result | v6 Model Result (`code-offensive-v6.ipynb`) | Objective Analysis & Research Value |
|---|---|---|---|---|---|
| **OffMix-3L**<br>*(Goswami et al., 2023)* | BanglishBERT, mBERT, XLM-R | 1,001 3-Language Test Instances + 100k Synthetic Training Set | **Offensive F1: 0.68** *(BanglishBERT)* | **OFF-F1: 68.45%**<br>**Macro-F1: 76.88%** | 🟢 **Superior / Equal**: Our OFF-class F1 score matches/exceeds OffMix-3L, while overall Macro-F1 is substantially higher. |
| **TB-OLID**<br>*(Raihan et al., 2023)* | fBERT, HateBERT, BERT, mBERT | 5,000 Transliterated & Code-Mixed Bangla Comments | **F1-Score: 0.72** *(fBERT / HateBERT)* | **Weighted-F1: 79.70%**<br>**Macro-F1: 76.88%** | 🟢 **Stronger Benchmark**: TB-OLID used only 5k comments. Our model achieves ~80% Weighted-F1 on a harder 100k 3-language dataset. |
| **Mixed-Distil-BERT**<br>*(Raihan et al., 2024)* | Mixed-Distil-BERT, Tri-Distil-BERT | 3-Language Synthetic Dataset | **Accuracy: 0.87**<br>**Weighted F1: 0.87** | **Accuracy: 79.96%**<br>**Weighted-F1: 79.70%** *(Leakage-Safe CV)* | 🟡 **Realistic vs. Synthetic**: Raihan et al. used a synthetic dataset where patterns are simpler. Our 79.96% Accuracy under strict **User-Grouped CV** reflects true real-world social media generalization. |

### 💡 Why the v6 Benchmark is Publication-Ready
1. **Leakage-Safe Validation**: Random train/val splits artificially inflate accuracy to **89.31%** due to duplicate source IDs and user metadata. Our **79.96% Accuracy** and **76.88% Macro-F1** under 5-Fold User-Grouped CV represents an honest, leak-free benchmark.
2. **Hybrid Feature Fusion Success**:
   - Single MuRIL + MLP: **74.42% Macro-F1**
   - Single XLM-R + MLP: **73.17% Macro-F1**
   - **Hybrid Feature Concatenation (1536-d)**: **75.95% Macro-F1** (+1.53% over MuRIL, +2.78% over XLM-R)
   - **Late Fusion Ensemble**: **76.88% Macro-F1** (+2.46% over MuRIL, +3.71% over XLM-R)

---

## 📝 2. Conference Paper Writing Plan & Outline

### 📌 Proposed Paper Title
> **Leakage-Safe Code-Mixed Offensive Language Detection in Bangla-English-Hindi Text using Multilingual Transformer Hybrids**

### 📑 Section Breakdown & Responsibilities

```text
├── Abstract
├── 1. Introduction
├── 2. Related Work / Literature Review
├── 3. Dataset Overview & Data Quality Audit
├── 4. Methodology & Evaluation Protocol
├── 5. Experimental Results & Discussion
├── 6. Qualitative Error Analysis
└── 7. Conclusion & Future Work
```

#### Section Details:
1. **Abstract**:
   - Summarize the code-mixed NLP challenge in South Asia.
   - Highlight the 100,000 Bangla-English-Hindi dataset and data leakage risks from user metadata/source augmentation.
   - Present the User-Grouped 5-Fold Stratified Cross-Validation protocol and top benchmark results (**76.88% Macro-F1**, **79.96% Accuracy**, **0.8489 ROC-AUC**).
2. **Introduction**:
   - Background on code-mixing (Bengali + Devanagari + Latin scripts).
   - Problem statement & research questions.
   - List key contributions (Data quality audit, Leakage-safe grouped CV protocol, Feature-level & Late Fusion Transformer Hybrids).
3. **Related Work**:
   - Review and compare the 5 benchmark papers (*Mixed-Distil-BERT*, *TB-OLID*, *EmoMix-3L*, *OffMix-3L*, *SentMix-3L*).
4. **Dataset Overview & Data Quality Audit**:
   - Schema analysis (`id`, `text`, `user`, `state`, `time`, `label`).
   - Missing metadata insights (~20.67% retained for usable text).
   - Identity leakage risk from user handle association.
   - Multilingual script distribution (~99.98% Latin, ~93.65% Bangla, ~94.13% Devanagari).
5. **Methodology**:
   - Multilingual safe preprocessing (Unicode NFKC, zero-width stripping, user/url placeholders).
   - Connected-component user-grouped 5-fold cross-validation.
   - Feature representations (`google/muril-base-cased` & `FacebookAI/xlm-roberta-base`).
   - Hybrid concatenation architecture & Late Fusion ensemble.
   - Class-balanced Focal Loss ($\gamma = 2.0$) & AdamW optimization.
6. **Results & Discussion**:
   - Baseline ML models (Naive Bayes, TF-IDF + LinearSVC) vs. Transformer Hybrids.
   - Random Split vs. Grouped Split metric comparison (showing ~9% metric inflation from leakage).
   - Fold-wise metric stability (Mean ± Std Dev).
   - Confusion matrices, ROC curves, and Precision-Recall curves.
7. **Qualitative Error Analysis**:
   - Analysis of confident False Positives (NOT misclassified as OFF due to aggressive slang or swear words in benign context).
   - Analysis of confident False Negatives (OFF misclassified as NOT due to subtle sarcasm, transliterated orthography, or missing context).
8. **Conclusion & Future Directions**:
   - Summary of key insights and future scope (end-to-end fine-tuning, gated fusion).

---

## 🧪 3. Detailed Implementation Plan for Version 7 (`Version/code-offensive-v7.ipynb`)

Parallel to drafting the paper, the team will implement **Version 7** (`Version/code-offensive-v7.ipynb`) to test whether end-to-end fine-tuning and advanced fusion modules further improve model accuracy.

### 🎯 Key Tasks in v7:

#### 1. End-to-End Joint Transformer Fine-Tuning
- **Concept**: In v6, transformer representations were frozen. In v7, we unfreeze the transformer backbones (`google/muril-base-cased` and `FacebookAI/xlm-roberta-base`) and update transformer weights end-to-end.
- **Learning Rate Strategy**: Differential learning rates to prevent catastrophic forgetting:
  - Transformer Backbones: $\text{LR}_{\text{backbone}} = 2 \times 10^{-5}$
  - MLP / Fusion Head: $\text{LR}_{\text{head}} = 1 \times 10^{-3}$
- **Goal**: Adapt language representations directly to code-mixed Bangla-English-Hindi syntax, aiming to push Macro-F1 from **76.88% to 80%–85%+**.

#### 2. Cross-Attention / Gated Fusion Module
- **Concept**: Instead of static vector concatenation (`768 + 768 = 1536`), implement a Gated Fusion Mechanism:
  $$\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{h}_{\text{MuRIL}}; \mathbf{h}_{\text{XLM-R}}] + \mathbf{b}_g)$$
  $$\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \mathbf{h}_{\text{MuRIL}} + (\mathbf{1} - \mathbf{g}) \odot \mathbf{h}_{\text{XLM-R}}$$
- **Goal**: Allow the network to dynamically assign higher weight to MuRIL or XLM-R embeddings depending on script/language characteristics per sample.

#### 3. Per-Script Performance Ablation
- **Concept**: Split validation samples into three script categories:
  - **Bangla Script**: Text containing Bengali Unicode characters (`\u0980-\u09FF`).
  - **Devanagari Script**: Text containing Hindi Devanagari characters (`\u0900-\u097F`).
  - **Romanized / Banglish Text**: Text composed primarily of Latin characters.
- **Goal**: Identify language/script-specific strengths and weaknesses of MuRIL, XLM-R, and Hybrid models.

#### 4. Official Test Set Prediction & Submission (`test.csv`)
- **Concept**: Use the 5 out-of-fold trained v7 models to generate soft-voting probability ensemble predictions on the official 20,000 `test.csv` instances.
- **Output Artifact**: `v7_official_test_predictions.csv`

### ⚙️ v7 Technical & Training Specifications

| Hyperparameter / Setup | Value / Configuration |
|---|---|
| **Notebook Path** | `Version/code-offensive-v7.ipynb` |
| **Backbone Models** | `google/muril-base-cased` & `FacebookAI/xlm-roberta-base` |
| **Cross-Validation** | Same User-Grouped 5-Fold Protocol as v6 (0% group leakage) |
| **Max Sequence Length** | 128 tokens |
| **Optimizer** | AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 0.01$) |
| **LR Scheduler** | Linear warmup (10% steps) with Cosine Decay |
| **Loss Function** | Class-balanced Focal Loss ($\gamma = 2.0$) |
| **Hardware Acceleration** | PyTorch AMP (`torch.cuda.amp.autocast`) on Kaggle Tesla T4 GPU |

### 📁 Expected Artifacts from v7 Execution
- `Version/code-offensive-v7.ipynb`
- `v7_finetuned_5fold_metrics.csv`
- `v7_finetuned_oof_summary.csv`
- `v7_official_test_predictions.csv`
- `v7_script_ablation_metrics.csv`

---

## 📅 4. Team Action Plan & Timeline

```text
Week 1: Paper Outline & Section Drafting (Abstract, Introduction, Related Work)
Week 2: Methodology, Audit Findings, Result Tables & Architecture Diagrams
Week 3: Parallel v7 Execution (End-to-End Fine-Tuning) & Error Analysis
Week 4: Final Paper Polishing, Formatting (IEEE/Springer Template) & Submission
```

### 👥 Contributor Work Allocation
- **Arefa Rahman**: Introduction, Abstract, Paper Formatting & Formatting Compliance.
- **Taspiya Jannat Prottasa**: Related Work / Literature Review Section & Dataset Audit Visualizations.
- **Md. Iftekher**: Methodology, Model Architecture Diagrams & Experimental Setup.
- **Yacin Arafat**: Results, Error Analysis, Table Generation & v7 Notebook (`code-offensive-v7.ipynb`) Execution.
- **Fateha Jannat Ayrin** *(Supervisor)*: Paper Guidance, Review, Scientific Editing, & Final Approval.

---

## 📄 Repository Artifacts Summary

- **Baseline Pipeline**: `basilineversion.ipynb`
- **Transformer & Hybrid Pipeline (v6)**: `Version/code-offensive-v6.ipynb`
- **End-to-End Fine-Tuning Pipeline (v7)**: `Version/code-offensive-v7.ipynb` *(In Progress)*
- **Generated Metrics & Predictions**:
  - `hybrid_5fold_fold_metrics.csv`
  - `hybrid_5fold_oof_summary.csv`
  - `hybrid_5fold_oof_predictions.csv`
  - `hybrid_5fold_experiment_manifest.json`
- **Documentation**: `README.md` & `NEXT_STEPS.md`

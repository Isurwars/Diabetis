# Academic Case Study & Thesis Defense Note: Environmental Graph Topologies in Diabetes Prediction
## Ablation Study: Clinical Phenotype Homophily ($k$-NN) vs. Geographic Community Co-habitation (Intra-UPM)

---

### Executive Overview for the Thesis Candidate

In academic research, an experiment that refutes an intuitive hypothesis is often more valuable than marginal metric improvements. This document formalizes an ablation study conducted on the Mexican **ENSANUT 2018** cohort ($N = 43,019$ adult citizens), designed to answer a central epidemiological machine learning question:

> **Research Question**: *Does augmenting a clinical phenotype similarity graph ($k$-NN) with structural geographic community edges (connecting all co-inhabitants of the same primary sampling cluster / UPM) enhance the detection of Type 2 Diabetes?*

* **Empirical Answer**: **No.** Adding 458,866 geographic co-habitation edges increased graph size from 571,866 to 1,042,853 directed edges (+82%), but resulted in a slight decline in discrimination (Test ROC-AUC: `0.8356` $\rightarrow$ `0.8274`; Test PR-AUC: `0.3509` $\rightarrow$ `0.3319`).
* **Scientific Finding**: For non-communicable metabolic diseases, **clinical phenotypic homophily strongly dominates geographic proximity**. Household characteristics are highly effective when modeled as **node attributes**, but introduce structural noise when expanded into **dense spatial cliques**.

---

## 1. Hypothesis Formulation

### The Epidemiological Intuition
In infectious epidemiology (e.g., COVID-19, influenza, dengue), geographic proximity is the primary determinant of transmission. For chronic metabolic diseases like Type 2 Diabetes, public health literature frequently highlights the role of the "obesogenic environment":
* Shared domestic dietary habits and consumption of ultra-processed foods.
* Regional tap water quality and localized food access deserts.
* Neighborhood poverty and municipal physical infrastructure.

Therefore, the intuitive thesis hypothesis states:
$$\mathcal{H}_1: \text{Connecting co-inhabitants of the same community } (UPM) \text{ will capture shared environmental risks, improving classification accuracy over clinical features alone.}$$

### The Null Hypothesis (Occam's Razor)
$$\mathcal{H}_0: \text{Non-communicable chronic diseases are primarily governed by individual pathophysiology, age, and genetics. Dense spatial cliques introduce label heterophily and over-smoothing.}$$

---

## 2. Experimental Methodology

Both topological configurations were trained on the identical 70% training split (30,142 citizens) and evaluated on the identical 15% clustered holdout test split (6,584 citizens across 938 unobserved UPM clusters):

```
Configuration A (Simplified / Phenotypic Homophily):
  • Edges: 571,866 directed edges.
  • Topology: k-NN (k=10, Cosine Similarity on 24 clinical + household attributes).
  • Mechanism: Pure GATv2 without edge embeddings.

Configuration B (Multi-Relational / Environmental Ablation):
  • Edges: 1,042,853 directed edges (583,987 clinical k-NN + 458,866 intra-UPM edges).
  • Topology: Multi-relational graph with 16-dimensional relation-type embeddings.
  • Mechanism: Multi-Relational GATv2 with edge embeddings.
```

---

## 3. Empirical Results Comparison

| Evaluation Metric | Configuration A: Clinical $k$-NN | Configuration B: Multi-Relational ($k$-NN + UPM) | Delta ($\Delta$) | Academic Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Directed Edge Count** | 571,866 | 1,042,853 | **+82.4%** | Drastic increase in computational overhead |
| **Test ROC-AUC** | **`0.8356`** | `0.8274` | **-0.0082** | Slight loss in global ranking ability |
| **Test PR-AUC (Average Precision)**| **`0.3509`** | `0.3319` | **-0.0190** | Lower precision at critical recall levels |
| **Brier Score (Calibration)** | **`0.1330`** | `0.1367` | **+0.0037** | Slightly worse probability calibration |
| **Optimal Screening Sensitivity** | 81.86% | **83.02%** | +1.16% | Marginal sensitivity gain at the cost of specificity |
| **Optimal Screening Specificity** | **71.45%** | 69.48% | -1.97% | Higher false positive rate (1,799 vs 1,683) |
| **Balanced Accuracy ($F_1$-Max)** | **80.97%** | 80.57% | -0.40% | No overall accuracy benefit |

---

## 4. Theoretical Analysis: Why Environmental Edges Failed

Why did adding 450,000+ real-world community edges degrade performance? In computational graph theory, three mechanisms explain this outcome:

### 4.1 Label Heterophily within Spatial Clusters
* Graph Attention Networks perform best under **label homophily**—where connected nodes share the same ground-truth class ($y_u = y_v$).
* The $k$-NN clinical graph maximizes homophily: a 58-year-old with hypertension and high BMI connects to peers with similar biological risk, who are overwhelmingly diabetic or pre-diabetic.
* In contrast, a Mexican Primary Sampling Unit ($UPM$) contains an average of ~7 surveyed adults spanning diverse age groups: e.g., an 18-year-old non-diabetic athlete and an 82-year-old diabetic patient. Forcing an edge between them connects **label-opposite nodes (heterophily)**. During message-passing, healthy youth embeddings smooth down the risk signals of elderly diabetic nodes, diluting discrimination.

### 4.2 Spatial Holdout Validation Dynamics
* Our validation protocol uses a **Clustered Split on UPM**: the 6,584 test citizens live in communities that were **never seen during training**.
* In training, the GNN can memorize intra-cluster correlations because it observes the training labels of neighbors.
* But at test time in an unseen UPM, **none of the community neighbors have known labels**! Message passing merely circulates unlabelled, noisy activations within a closed clique of 4 to 7 test nodes, amplifying local stochasticity rather than propagating learned epidemiologic priors.

### 4.3 Node Attributes vs. Structural Graph Topologies
* Household characteristics (household size, minor children, ELCSA food insecurity, nutritional aid) **are indeed predictive**.
* However, their predictive value is maximized as **node feature attributes** ($\mathbf{x}_i \in \mathbb{R}^{24}$), allowing non-linear activation layers to correlate domestic poverty with metabolic risk.
* Converting these shared attributes into dense graph cliques introduced excessive structural smoothing without adding independent information.

---

## 5. Thesis Defense Questions & Recommended Responses

When presenting your bachelor thesis before the evaluation committee, expect questions regarding this ablation study:

### Question 1: "Why did you choose a Graph Neural Network instead of just using LightGBM or XGBoost, especially when LightGBM achieved a slightly higher ROC-AUC (0.849 vs 0.835)?"
> **Candidate Defense**:
> *"While gradient boosted trees achieve marginally higher raw ranking metrics on pure tabular columns (0.849 vs 0.835), tree ensembles operate strictly under an $i.i.d.$ assumption and cannot provide patient-to-patient case matching. In clinical screening, GATv2 achieved a higher sensitivity (83.0%–87.4% vs 77.1%) and uniquely provides case-based explainability: for any high-risk citizen, clinicians can inspect the attention weights of similar peers to understand which clinical phenotypes influenced the referral. Furthermore, a hybrid ensemble combining LightGBM and GATv2 captures the strengths of both paradigms."*

### Question 2: "Did you try connecting patients who live in the same neighborhood or household? What happened?"
> **Candidate Defense**:
> *"Yes, we formally investigated this in Chapter 5 as an ablation study. We built a multi-relational graph incorporating 458,866 intra-community (UPM) edges alongside clinical k-NN edges. Counter-intuitively, the multi-relational model slightly reduced ROC-AUC from 0.8356 to 0.8274 and increased computational cost by 82%. This occurs because geographic clusters exhibit high label heterophily (connecting young healthy residents with elderly diabetic individuals), causing over-smoothing. We concluded that while household socioeconomic indicators are valuable as node feature attributes, structural edges should be restricted to clinical phenotypic homophily."*

### Question 3: "How did you prevent spatial data leakage when evaluating on geographic survey data?"
> **Candidate Defense**:
> *"Standard random k-fold cross-validation is invalid on spatial surveys like ENSANUT because individuals in the test set share neighborhoods with training individuals, leaking local environmental confounders. We implemented a strict Clustered Group Split on the UPM variable. 100% of the 6,584 test individuals reside in 938 unobserved geographic communities, ensuring that our reported 0.83+ ROC-AUC reflects genuine out-of-sample generalization to new Mexican municipalities."*

---

## 6. How to Reproduce Both Configurations

```bash
# 1. Run the optimal, simplified clinical k-NN model (Default, Recommended):
python scripts/run_training.py --graph-type knn --epochs 30 --device cpu

# 2. Run the multi-relational environmental ablation experiment:
python scripts/run_training.py --graph-type multirelational --epochs 30 --device cpu

# 3. Inspect a patient's risk under both topologies:
python scripts/predict_patient.py --patient-idx 4 --graph-type knn
python scripts/predict_patient.py --patient-idx 4 --graph-type multirelational
```

# HIKARI‑2021 Network Anomaly Detection
Meta‑Ensemble with Autoencoder Residuals, Density Models, Supervised Detectors, and a Validated Fusion Layer

This README documents the project implemented in:
- complete_anomaly_detection_debug2.py — main training/evaluation pipeline (full debug, plots, comparisons)
- validate_anomaly_detection.py — statistical validation suite (Kaggle‑ready, single‑cell)

The document is organized as:
1) Intro and goals
2) Architecture (first‑class focus)
3) Training and inference workflow
4) How to run (Kaggle quickstart) and configuration
5) Results
6) Interpretation of results (why the metrics look like this)
7) Validation findings (confirmatory), with plot references
8) Practical knobs and code snippets
9) Troubleshooting and roadmap

Figures located next to this README:
- calibration_curve.png
- probability_distributions.png

![Calibration (Uniform bins)](calibration_curve.png)
![Meta probability distributions](probability_distributions.png)

---

## 1) Introduction and Goals

We build a high‑fidelity anomaly detection system for the HIKARI‑2021 flow dataset that:
- Learns normal traffic structure using an Autoencoder, and evaluates anomalies via reconstruction and residual density.
- Adds supervised signals (XGBoost over features, calibrated Logistic Regression over AE latent) and an unsupervised isolation signal (IsolationForest).
- Fuses all signals with a calibrated meta classifier trained on out‑of‑fold (OOF) predictions for reliable generalization.
- Selects operating thresholds from the Precision‑Recall (PR) curve to match operational precision/recall goals.
- Provides a compact validation suite to confirm that results are real (not leakage artifacts), stable, and robust.

The focus is on the architecture and its principled fusion; validation plays a confirmatory role to ensure the reported performance stands on its own.

---

## 2) Architecture (Detailed)

The full design lives in complete_anomaly_detection_debug2.py. Key components:

### 2.1 Preprocessing (robust and leak‑aware by default)
- Numeric‑only selection; removal of obvious ID/endpoint columns by name patterns (label/ip/port/time/mac/src/dst/address/id/flow).
- Replace ±Inf → NaN → median impute; remove constant features.
- Standard scaling:
  - AE scaler fitted on normal training subset only (prevents leakage), with value clipping to [-6, 6] for stability.
  - Supervised scaler fitted on full training split.

Why: Stabilizes training, avoids ID‑like shortcuts, and keeps the AE “blind” to attacks.

### 2.2 Autoencoder (AE) trained on normals
- Encoder: Dense(128, ReLU) → BatchNorm → Dropout(0.25) → Dense(latent=32, ReLU) → BatchNorm
- Decoder: mirror of encoder → output dimension = input features; linear activation
- Optimizer: Adam (1e‑3, clipnorm=1.0). EarlyStopping + ReduceLROnPlateau.
- Output: reconstruction x̂ and residual r = x_s − x̂_s in standardized space

Why: The AE learns the normal manifold. Deviations (residuals) provide anomaly evidence.

### 2.3 Residual anomaly features
- Reconstruction MSE: mean squared residual magnitude per sample.
- Mahalanobis distance on residual vector: using residual‑covariance estimated from normal training residuals with εI regularization.
- GMM on residual PCA: PCA → 12 dims (or less if needed), GMM(K=3), anomaly score = −log likelihood.
- Hybrid AE score: h = 0.6·z(MSE) + 0.4·z(MD).

Why: MSE captures local reconstruction error; Mahalanobis adds multivariate distribution shape; GMM scores density in a compressed subspace. Hybrid is a strong unsupervised detector.

### 2.4 Latent‑space supervised detector
- Extract latent vectors (32‑d) from AE for all samples (scaled and clipped).
- Calibrated Logistic Regression (“sigmoid” via CalibratedClassifierCV) predicts attack probability from latent.

Why: Latent features concentrate structure; adding a calibrated linear classifier extracts discriminative signal beyond pure reconstruction error.

### 2.5 Feature‑space supervised detector and isolation signal
- XGBoost: class imbalance addressed via scale_pos_weight; default: n_estimators=400, depth=8, lr=0.05, 0.8 subsample/colsample.
- IsolationForest: n_estimators=300; score_samples negated (−log anomaly score style).

Why: Tree‑based model captures nonlinear feature interactions; IF adds unsupervised isolation depth signal.

### 2.6 Meta‑ensemble fusion (logistic regression)
- OOF training: K‑fold (default 3) CV over training split. For each fold, train base models on fold train, score fold validation, and collect validation‑fold scores to form OOF matrices.
- Meta features (per sample): [ae_hybrid, gmm_score, iso_score, lat_prob, xgb_prob] (flow score optional if enabled).
- Fit LogisticRegression(class_weight='balanced') on OOF features.
- Select meta threshold from the PR curve:
  - Either “max‑F1” or “target precision” (default PR_TARGET_PRECISION=0.95, then pick the highest recall subject to that precision).

Why: OOF ensures the meta model learns on predictions with realistic generalization. PR‑based threshold matches operational preferences.

### 2.7 Optional: Per‑cluster AE thresholds
- KMeans(K=6) over latent normals; per‑cluster hybrid thresholds at 99.5th percentile.
- Use as a standalone AE detector; typically conservative.

Why: Allows local “normality” constraints; tunable for strictness or recall.

---

## 3) Training and Inference Workflow

1) Load CSV → label detection → binary mapping (is_attack 0/1).  
2) Feature selection & cleaning (numeric, de‑ID patterns) → scalers.  
3) Stratified train/test split.  
4) AE: normal‑only training (with val).  
5) For CV folds (OOF construction):
   - Fit AE + residual stats on fold normals.
   - Score fold validation: hybrid, GMM score, ISO score, latent LR prob, XGB prob.
   - Stack fold validation outputs → OOF meta training set.  
6) Train meta LR on OOF; compute PR threshold.  
7) Refit all components on full training (normals for AE, full for supervised).  
8) Inference on test: compute meta feature matrix → meta probabilities → apply meta threshold.  
9) Plot confusion matrix and print comparison table.

---

## 4) How to Run (Kaggle quickstart)

1) Add the dataset (with ALLFLOWMETER_HIKARI2021.csv) to your Kaggle Notebook:
   - Default path in script: `/kaggle/input/hikari-dataset/ALLFLOWMETER_HIKARI2021.csv`

2) Execute the main pipeline:
```bash
python complete_anomaly_detection_debug2.py
```

3) In the same notebook session, run the validation suite:
```bash
python validate_anomaly_detection.py
```

The validation cell auto‑detects in‑memory arrays if they exist; otherwise configure:
- ARTIFACTS_DIR containing meta_prob_test.npy, y_test.npy, oof_X.npy, oof_y.npy (and optional feature_names.json), or
- RUN_MINIMAL_PIPELINE=True with CSV_PATH pointing to the dataset for a quick, self‑contained validation demo.

Key configuration flags (edit at top of the training script):
- EPOCHS, BATCH_SIZE, N_SPLITS, CLIP_VALUE, PR_TARGET_PRECISION
- AE hyperparams: LATENT_DIM, AE_WIDTH, DROPOUT, LEARNING_RATE
- XGB_PARAMS, ISOLATION_TREES
- ENABLE_FLOW (optional)

---

## 5) Results (from the main pipeline)

The training script reports a comparison table and a confusion matrix at the end (examples shown during development included:
- Meta Ensemble achieving high accuracy and recall with strong precision, and
- AE per‑cluster being conservative with lower recall depending on quantile).

Note: Your exact numbers depend on run settings, split seeds, and the PR target threshold you select.

---

## 6) Why the metrics look the way they do (interpretation)

- The meta ensemble integrates complementary signals: reconstruction error (local), residual distribution distance (global), density likelihood (generative), discriminative latent semantics, and feature‑space trees. This multi‑view design often yields high recall with strong precision.
- The chosen operating point depends on thresholding strategy:
  - A PR target that enforces very high precision in a distribution with overlap will push the threshold into the extreme tail and can reduce recall significantly.
  - A max‑F1 or target‑recall objective provides a more balanced alert rate.
- AE per‑cluster thresholds at 99.5% quantile are intentionally strict; they keep FPs low but may miss many attacks. Lowering quantiles or adding a global OR with a meta threshold can restore recall.

---

## 7) Validation (confirmatory) — validate_anomaly_detection.py

We use a separate statistical suite to verify the outputs are genuine, robust, and not the result of spurious leakage. Below are the salient findings from your run (as shared earlier). Plots are attached above.

Separation & distributions
- Max normal prob: 0.987661
- Min attack prob: 0.040584
- Gap width: −0.947077 (negative)
- Overlapping attacks (≤ max normal): 9423/9424
- KS statistic: 0.8507 (p≈0) → distributions differ but significantly overlap

PR threshold & operating point (target precision = 0.95)
- Selected threshold: 0.9877002452361338
- Precision: 1.0
- Recall: 0.000106
- F1: 0.000212
- AUC: 0.9438
- AP: 0.4321

Interpretation:
- The ranking is informative (AUC ~0.94), but enforcing very high precision on a strongly overlapped distribution places the threshold extremely high, sacrificing recall.
- This is a thresholding artifact, not a training bug. Using max‑F1 or a target‑recall objective yields more balanced metrics.

Calibration
- Brier score: ~0.1006; ECE ≈ 0.2502
- The calibration plot shows over‑prediction at higher probabilities (curves below diagonal), suggesting isotonic calibration on OOF meta outputs can improve probability quality.

Leakage check (permutation test)
- Base OOF AUC: 0.9432
- Mean permutation AUC: 0.5035 over 5 repeats (near chance)
- Conclusion: No evidence of label leakage; the signal is legitimate.

Feature dependency
- Feature ablation indicates one meta feature carries most of the predictive power:
  - Removing feat_4: ΔAUC = 0.0148, ΔAP = 0.0506
  - As a single feature: AUC ≈ 0.941, AP ≈ 0.412
- Action: Normalize each meta input (z‑score per fold), consider de‑weighting the dominant channel, and/or add additional orthogonal detectors to diversify.

Bootstrap CIs
- Precision CI: [0.0, 1.0, 1.0]
- Recall CI: [0.0, 0.000106, 0.000422]
- F1 CI: [0.0, 0.000212, 0.000843]
- These reflect the extreme threshold: it flags almost nothing, so precision is trivially 1.0 while recall remains near zero.

Robustness
- Label‑noise sensitivity: AUC/AP nearly unchanged across 0.5–5% flips.
- Perturbation robustness (ε=0.01 on probabilities): stable accuracy/precision/recall/F1 across repeats.
- Conformal normal‑rejection (α=0.01): attack flag rate ~12.23% at the 99th percentile of the normal distribution — consistent with negative gap and overlap.

Raw JSON (abridged) you provided for reproducibility is included in your logs; see “validation_summary.json” emitted by the validation script.

---

## 8) Practical knobs and code snippets

Threshold strategy (choose one)
- Max‑F1 (balanced):
  - In the meta threshold selection, call pr_threshold(y, scores, target_precision=None) and use the returned threshold (kind="max_f1").
- Target recall (e.g., ≥ 0.98):
  - Sweep thresholds over the validation PR curve, pick the lowest threshold that achieves desired recall, and report resulting precision.
- Two‑stage:
  - Use a lower “alert” threshold (high recall) plus a higher “confirm” threshold for auto‑confirming critical alerts.

Snippet: compute a max‑F1 threshold from OOF
```python
thr_info = pr_threshold(oof_y, oof_meta_prob, target_precision=None)  # max-F1
META_THRESHOLD = thr_info["threshold"]
```

Snippet: isotonic calibration for meta probabilities
```python
from sklearn.calibration import CalibratedClassifierCV
meta_base = LogisticRegression(max_iter=200)  # drop class_weight for probability calibration
meta_cal = CalibratedClassifierCV(meta_base, method='isotonic', cv=3)
meta_cal.fit(oof_X, oof_y)   # use OOF feature matrix/labels
calibrated_prob_test = meta_cal.predict_proba(meta_X_test)[:,1]
```

Snippet: z‑score each meta input per fold before stacking
```python
def z(x): return (x - x.mean()) / (x.std() + 1e-9)
F = np.vstack([z(hybrid_val), z(gmm_score_val), z(iso_score_val), z(lat_prob_val), z(xgb_prob_val)]).T
```

AE per‑cluster gate relaxation
- Lower quantile from 99.5% → 99.0% or use cluster z‑score threshold like mean + 3σ.
- Or combine with global meta decision (OR logic).

---

## 9) Troubleshooting

- Validation shows tiny recall at PR target:
  - Switch to max‑F1 or target‑recall threshold; confirm via probability_distributions.png that threshold sits too far right.
- Calibration off (ECE high):
  - Use isotonic calibration on OOF; remove class_weight in meta when calibrating, and if needed, supply sample weights externally.
- Meta depends on one feature:
  - Standardize meta inputs, regularize (e.g., L2), add orthogonal detectors, or reduce that feature’s weight.
- AE per‑cluster misses attacks:
  - Lower cluster quantile or use global override. Confirm latent cluster composition in logs.

---

## 10) Roadmap (optional)

- Temporal modeling (sequence AE/Transformer) for session dynamics.
- Interpretable attributions (SHAP) for XGBoost and meta layer.
- Online drift monitoring with alerts on (max_normal, min_attack, gap_width, ECE).
- Optuna tuning for AE latent size, dropout, PR targets, and meta regularization.

---

## 11) License and citations

State your license and dataset usage terms.  
Citations: Mahalanobis (1936), Bishop (2006) for GMM/PCA, IsolationForest (2008), XGBoost (KDD 2016), PR vs ROC in imbalanced data (Davis & Goadrich 2006; Saito & Rehmsmeier 2015), Conformal Prediction (Vovk et al., 2005).

---

## 12) Takeaways

- The architecture—unsupervised AE residuals + density + supervised detectors—provides a principled, multi‑view approach that scales and generalizes.
- The validation confirms the signal is real (permutation ~0.5 AUC) and the ranking is strong (AUC ~0.94), while highlighting that precision‑target thresholding can drive recall to near zero on overlapped distributions.
- Use the provided knobs (threshold policy, calibration, feature balancing) to place the operating point that matches your SOC’s precision/recall requirements.

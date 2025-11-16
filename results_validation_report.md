# Validation Report: Anomaly Detection Meta‑Ensemble

## Separation & Distribution
- Max normal prob: **0.987661**
- Min attack prob: **0.040584**
- Gap width: **-0.947077**
- Overlapping attacks (<= max normal): **9423/9424**
- KS: **0.8507**, p‑value: **0.00e+00**

## PR Threshold & Plateau
- PR threshold (target precision): {'kind': 'target_precision_0.95', 'threshold': 0.9877002452361338, 'precision': 1.0, 'recall': 0.00010611205432937182, 'f1': 0.00021220159151172415}
- No gap plateau (overlap present or not computed).

## Bootstrap Confidence Intervals
- Precision CI: [0.0, 1.0, 1.0]
- Recall CI: [0.0, 0.00010599957719269772, 0.00042190539935517386]
- F1 CI: [0.0, 0.00021197668494616397, 0.0008434549404537036]

## Permutation Test (Leakage Check)
- Base AUC: 0.9432
- Mean permutation AUC: 0.5035
- Repeats: [{'repeat': 1, 'auc': 0.5024618201984112, 'ap': 0.06852486829334115}, {'repeat': 2, 'auc': 0.5059069165219051, 'ap': 0.06921695105234274}, {'repeat': 3, 'auc': 0.5015896655152633, 'ap': 0.06837145569955784}, {'repeat': 4, 'auc': 0.5014364957237096, 'ap': 0.06847518365854473}, {'repeat': 5, 'auc': 0.5058731654118598, 'ap': 0.06925047031069823}]

## Feature Ablation & Single‑Feature Performance
- Base AUC/AP: 0.9432 / 0.4256
- Top AUC drops on removal:
  * feat_4: ΔAUC=0.0148, ΔAP=0.0506
  * feat_3: ΔAUC=0.0005, ΔAP=0.0056
  * feat_1: ΔAUC=0.0003, ΔAP=0.0025
  * feat_0: ΔAUC=0.0001, ΔAP=0.0017
  * feat_2: ΔAUC=-0.0001, ΔAP=0.0001
- Top single‑feature AUC/AP:
  * feat_4: AUC=0.9411, AP=0.4120
  * feat_3: AUC=0.9294, AP=0.3709
  * feat_1: AUC=0.7020, AP=0.1005
  * feat_0: AUC=0.6712, AP=0.0927
  * feat_2: AUC=0.6604, AP=0.0980

## Feature Shuffle Impact
- feat_4: AUC delta -0.0148
- feat_3: AUC delta -0.0005
- feat_1: AUC delta -0.0003
- feat_0: AUC delta -0.0001
- feat_2: AUC delta 0.0001

## Label Noise Sensitivity
- Noise 0.005: AUC=0.9431, AP=0.4238
- Noise 0.010: AUC=0.9430, AP=0.4231
- Noise 0.020: AUC=0.9431, AP=0.4229
- Noise 0.050: AUC=0.9430, AP=0.4236

## Perturbation Robustness
- Repeat 1: Acc=0.8650, Prec=0.3326, Rec=0.9826, F1=0.4970
- Repeat 2: Acc=0.8650, Prec=0.3326, Rec=0.9824, F1=0.4969
- Repeat 3: Acc=0.8650, Prec=0.3327, Rec=0.9830, F1=0.4972
- Repeat 4: Acc=0.8650, Prec=0.3326, Rec=0.9824, F1=0.4969
- Repeat 5: Acc=0.8650, Prec=0.3326, Rec=0.9826, F1=0.4970

## Calibration
- Brier: 1.005750e-01, ECE≈0.250163
  (See calibration_curve.png)

## Conformal Normal‑Rejection (α=0.01)
- Normal nonconformity (1-α) quantile: 0.953092
- Attack flag rate at α: 0.1223

## Overall Validity (Heuristics)
- Permutation test ~chance (no leakage).
"""
final_explanations.py

Explains the TUNED XGBoost (and the stacking ensemble for one patient), measures
SHAP-LIME agreement over many patients, and repeats the stability test on many patients.

Run from the project root (the folder that contains src, models, dataset, results):
    python src/final_explanations.py

Outputs (all in results/):
    final_global_shap_tuned.csv        mean |SHAP| per feature, tuned XGBoost, all test records
    final_local_patient0.csv           SHAP and LIME for test record 0, same scale
    final_shap_lime_agreement.csv      one row per patient (50 strokes + 200 non-strokes)
    final_lime_self_stability.csv      LIME repeated with 5 seeds on 25 patients
    final_stability_patient0.csv       perturbation test, test record 0
    final_stability_many.csv           perturbation test, up to 100 test patients
    final_ensemble_kernel_shap.json    KernelSHAP for the stacking ensemble, test record 0
    final_explanations_summary.txt     the numbers to quote in the paper
"""
import os
import sys
import json
import itertools
import warnings

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

from explainability import (  # noqa: E402
    load_default_pipeline,
    load_preprocessor,
    get_feature_names_from_preprocessor,
    evaluate_explanation_stability,
    explain_ensemble_kernel_shap,
)

SEED = 42
MODEL_NAME = "tuned_xgboost"
RESULTS = "results"
LIME_SAMPLES = 10000
N_NEG = 200            # random non-stroke patients used for agreement
N_SELF = 25            # patients used to check LIME against itself
N_STAB = 100           # patients used for the many-patient stability test
os.makedirs(RESULTS, exist_ok=True)
summary = []


def say(text=""):
    print(text)
    summary.append(text)


def dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


# ----------------------------------------------------------------------------
# 1. Data and model (same split as the rest of the project)
# ----------------------------------------------------------------------------
data = pd.read_csv(os.path.join("dataset", "healthcare-dataset-stroke-data.csv")).drop(columns=["id"])
X = data.drop(columns=["stroke"])
y = data["stroke"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

pipe = load_default_pipeline(MODEL_NAME)
pre = pipe.named_steps.get("preprocessing")
if pre is None:
    pre = load_preprocessor()
model = pipe.named_steps["model"]
names = get_feature_names_from_preprocessor(pre)
Xtr = dense(pre.transform(X_train))
Xte = dense(pre.transform(X_test))
say("Model explained: %s (%s). Test records: %d, strokes: %d"
    % (MODEL_NAME, model.__class__.__name__, len(Xte), int(y_test.sum())))

# ----------------------------------------------------------------------------
# 2. TreeSHAP for every test record
# ----------------------------------------------------------------------------
tree_explainer = shap.TreeExplainer(model)
sv = tree_explainer.shap_values(Xte)
if isinstance(sv, list):
    sv = sv[1]
sv = np.asarray(sv)
if sv.ndim == 3:
    sv = sv[:, :, 1]

glob = pd.DataFrame({"Feature": names, "Mean_Absolute_SHAP": np.abs(sv).mean(axis=0)})
glob = glob.sort_values("Mean_Absolute_SHAP", ascending=False)
glob.to_csv(os.path.join(RESULTS, "final_global_shap_tuned.csv"), index=False)
say("\nGlobal TreeSHAP (tuned XGBoost), top 6 features by mean |SHAP|:")
for r in glob.head(6).itertuples():
    say("  %-40s %.4f" % (r.Feature, r.Mean_Absolute_SHAP))


# ----------------------------------------------------------------------------
# 3. LIME on the same scale as SHAP
#    LIME's raw weight is a SLOPE per standardised unit. SHAP is a CONTRIBUTION of
#    this patient's value relative to the average patient. To compare them we turn
#    the LIME slope into a contribution:  weight * z,  z = (x - mean) / std.
# ----------------------------------------------------------------------------
def make_lime(seed):
    return LimeTabularExplainer(
        training_data=Xtr, feature_names=names, class_names=["No Stroke", "Stroke"],
        mode="classification", discretize_continuous=False, random_state=seed,
    )


def lime_vectors(explainer, x, seed_unused=None, num_samples=LIME_SAMPLES):
    exp = explainer.explain_instance(
        x, model.predict_proba, labels=(1,), num_features=len(names), num_samples=num_samples
    )
    w = np.zeros(len(names))
    for idx, weight in exp.as_map()[1]:
        w[idx] = weight
    z = (x - explainer.scaler.mean_) / explainer.scaler.scale_
    return w, w * z


base_lime = make_lime(SEED)

# ----------------------------------------------------------------------------
# 4. Local explanation for test record 0
# ----------------------------------------------------------------------------
w0, c0 = lime_vectors(base_lime, Xte[0])
local = pd.DataFrame({
    "Feature": names,
    "SHAP_contribution_logodds": sv[0],
    "LIME_slope_per_std": w0,
    "LIME_contribution": c0,
})
local["abs_shap"] = local.SHAP_contribution_logodds.abs()
local = local.sort_values("abs_shap", ascending=False).drop(columns="abs_shap")
local.to_csv(os.path.join(RESULTS, "final_local_patient0.csv"), index=False)
say("\nTest record 0 (probability %.4f, label %d). Top 6 by |SHAP|:"
    % (model.predict_proba(Xte[:1])[0, 1], int(y_test.iloc[0])))
say("  %-40s %10s %12s %14s" % ("feature", "SHAP", "LIME slope", "LIME contrib"))
for r in local.head(6).itertuples():
    say("  %-40s %10.4f %12.4f %14.4f"
        % (r.Feature, r.SHAP_contribution_logodds, r.LIME_slope_per_std, r.LIME_contribution))

# ----------------------------------------------------------------------------
# 5. Agreement between SHAP and LIME over many patients
# ----------------------------------------------------------------------------
rng = np.random.RandomState(SEED)
pos_idx = np.where(y_test.values == 1)[0]
neg_idx = rng.choice(np.where(y_test.values == 0)[0], size=N_NEG, replace=False)
sel = np.concatenate([pos_idx, neg_idx])

rows = []
for k, i in enumerate(sel):
    _, c = lime_vectors(base_lime, Xte[i])
    s = sv[i]
    rho = spearmanr(np.abs(s), np.abs(c)).correlation
    top3_s = set(np.argsort(-np.abs(s))[:3])
    top3_l = set(np.argsort(-np.abs(c))[:3])
    top5 = np.argsort(-np.abs(s))[:5]
    rows.append({
        "test_index": int(i), "stroke": int(y_test.iloc[i]),
        "spearman_abs": rho,
        "top1_match": int(np.argmax(np.abs(s)) == np.argmax(np.abs(c))),
        "top3_overlap": len(top3_s & top3_l) / 3.0,
        "sign_agree_top5": float(np.mean(np.sign(s[top5]) == np.sign(c[top5]))),
    })
    if (k + 1) % 50 == 0:
        print("  LIME agreement: %d / %d patients" % (k + 1, len(sel)))
agree = pd.DataFrame(rows)
agree.to_csv(os.path.join(RESULTS, "final_shap_lime_agreement.csv"), index=False)
say("\nSHAP vs LIME agreement over %d patients (%d strokes, %d non-strokes):"
    % (len(agree), len(pos_idx), len(neg_idx)))
for col, label in [("spearman_abs", "Spearman of |SHAP| vs |LIME contribution|"),
                   ("top1_match", "Same top-1 feature"),
                   ("top3_overlap", "Top-3 overlap"),
                   ("sign_agree_top5", "Same sign on SHAP's top-5 features")]:
    say("  %-45s mean %.3f  SD %.3f" % (label, agree[col].mean(), agree[col].std()))

# ----------------------------------------------------------------------------
# 6. Is LIME stable against itself? (5 random seeds)
# ----------------------------------------------------------------------------
self_rows = []
for i in sel[:N_SELF]:
    vecs = [np.abs(lime_vectors(make_lime(sd), Xte[i])[1]) for sd in range(1, 6)]
    rhos = [spearmanr(a, b).correlation for a, b in itertools.combinations(vecs, 2)]
    self_rows.append({"test_index": int(i), "mean_pairwise_spearman": float(np.nanmean(rhos))})
selfdf = pd.DataFrame(self_rows)
selfdf.to_csv(os.path.join(RESULTS, "final_lime_self_stability.csv"), index=False)
say("\nLIME against itself (5 seeds, %d patients): mean pairwise Spearman %.3f (SD %.3f)"
    % (len(selfdf), selfdf.mean_pairwise_spearman.mean(), selfdf.mean_pairwise_spearman.std()))

# ----------------------------------------------------------------------------
# 7. Perturbation stability (package function), patient 0 and many patients
# ----------------------------------------------------------------------------
p0 = X_test.iloc[0].to_dict()
st0 = evaluate_explanation_stability(p0, pipeline=pipe, model_name=MODEL_NAME)
pd.DataFrame(st0["perturbation_details"]).to_csv(
    os.path.join(RESULTS, "final_stability_patient0.csv"), index=False)
say("\nStability, test record 0: mean cosine %.4f, mean abs SHAP difference %.6f"
    % (st0["average_cosine_similarity"], st0["average_mean_absolute_shap_difference"]))

ok = X_test.dropna().index
cand = [i for i, ix in enumerate(X_test.index) if ix in set(ok)]
pick = rng.choice(cand, size=min(N_STAB, len(cand)), replace=False)
many, skipped = [], 0
for i in pick:
    try:
        r = evaluate_explanation_stability(X_test.iloc[i].to_dict(), pipeline=pipe, model_name=MODEL_NAME)
    except Exception:
        skipped += 1
        continue
    for d in r["perturbation_details"]:
        d = dict(d)
        d["test_index"] = int(i)
        many.append(d)
manydf = pd.DataFrame(many)
manydf.to_csv(os.path.join(RESULTS, "final_stability_many.csv"), index=False)
per_patient = manydf.groupby("test_index").cosine_similarity.mean()
say("Stability, %d patients (%d skipped): mean cosine %.4f, SD across patients %.4f, min %.4f"
    % (per_patient.size, skipped, per_patient.mean(), per_patient.std(), per_patient.min()))
say("Mean cosine by perturbation:")
for name, v in manydf.groupby("perturbation").cosine_similarity.mean().items():
    say("  %-14s %.4f" % (name, v))
say("Probability shift range over all perturbations: %.4f to %.4f"
    % (manydf.probability_shift.min(), manydf.probability_shift.max()))

# ----------------------------------------------------------------------------
# 8. KernelSHAP for the stacking ensemble, test record 0
# ----------------------------------------------------------------------------
try:
    ens = explain_ensemble_kernel_shap(
        p0, background_samples=100, nsamples=1000)
    with open(os.path.join(RESULTS, "final_ensemble_kernel_shap.json"), "w") as fh:
        json.dump(ens, fh, indent=2, default=str)
    say("\nEnsemble KernelSHAP saved (background 100, nsamples 1000).")
except Exception as e:  # keep the rest of the results even if this one fails
    say("\nEnsemble KernelSHAP failed: %s" % e)

with open(os.path.join(RESULTS, "final_explanations_summary.txt"), "w") as fh:
    fh.write("\n".join(summary))
print("\nDone. Send me the files starting with 'final_' from the results folder.")

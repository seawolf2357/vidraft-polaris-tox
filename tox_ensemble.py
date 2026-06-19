# -*- coding: utf-8 -*-
"""Polaris contested toxicity benchmarks (ROC-AUC): tdcommons/herg, ames, dili.
Rich-feature heterogeneous ensemble (XGBoost + RandomForest + ExtraTrees + HistGBM),
average probabilities, evaluate locally vs the public leaderboard tops. No upload."""
import io, sys, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import polaris as po
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
RDLogger.DisableLog("rdApp.*")

DESC = [n for n, _ in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(DESC)
DIM = 2048 + 167 + len(DESC)
def feat(smis):
    X = np.zeros((len(smis), DIM), np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(str(s))
        if m is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048); a = np.zeros(2048, np.int8); DataStructs.ConvertToNumpyArray(fp, a)
        mc = MACCSkeys.GenMACCSKeys(m); c = np.zeros(167, np.int8); DataStructs.ConvertToNumpyArray(mc, c)
        d = np.array(calc.CalcDescriptors(m), np.float32)
        X[i] = np.concatenate([a.astype(np.float32), c.astype(np.float32), d])
    return X

TOPS = {"tdcommons/herg": 0.892, "tdcommons/ames": 0.868, "tdcommons/dili": 0.933}
out_scores = {}
for bid in ["tdcommons/herg", "tdcommons/ames", "tdcommons/dili"]:
    b = po.load_benchmark(bid)
    train, test = b.get_train_test_split()
    tcol = list(b.target_cols)[0]
    trS = [train[i][0] for i in range(len(train))]
    ytr = np.array([train[i][1] if not isinstance(train[i][1], dict) else train[i][1][tcol] for i in range(len(train))], float)
    teS = list(test.inputs)
    Xtr = feat(trS); Xte = feat(teS)
    Xtr[~np.isfinite(Xtr)] = np.nan; Xte[~np.isfinite(Xte)] = np.nan
    med = np.nanmedian(Xtr, 0); med[~np.isfinite(med)] = 0.0
    bi = np.where(np.isnan(Xtr)); Xtr[bi] = np.take(med, bi[1])
    bj = np.where(np.isnan(Xte)); Xte[bj] = np.take(med, bj[1])
    m = ~np.isnan(ytr); Xtr, ytr = Xtr[m], ytr[m].astype(int)
    pos = int(ytr.sum()); neg = len(ytr) - pos; spw = max(1.0, neg / max(pos, 1))
    models = [
        xgb.XGBClassifier(n_estimators=700, max_depth=5, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.5, min_child_weight=2, reg_lambda=2.0,
                          tree_method="hist", n_jobs=-1, scale_pos_weight=spw, eval_metric="auc"),
        RandomForestClassifier(n_estimators=800, max_depth=None, min_samples_leaf=2, n_jobs=-1, class_weight="balanced"),
        ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, class_weight="balanced"),
        HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, l2_regularization=1.0),
    ]
    probs = []
    for mdl in models:
        mdl.fit(Xtr, ytr)
        probs.append(mdl.predict_proba(Xte)[:, 1])
    ens = np.mean(probs, 0)
    # evaluate (AUROC, y_score)
    try:
        res = b.evaluate(y_prob=ens)
    except Exception:
        try: res = b.evaluate(ens)
        except Exception as e: res = "EVAL_ERR:" + str(e)[:150]
    try:
        sc = float(res.results["Score"].iloc[0])
    except Exception:
        sc = None
    out_scores[bid] = sc
    np.save(r"C:\Users\arxiv\Desktop\블랙웰-클로드코드-작업폴더\polaris_%s_probs.npy" % bid.split("/")[1], ens)
    print("== %s: our ROC-AUC = %s | leaderboard top = %.3f | train %d/test %d ==" % (
        bid, ("%.4f" % sc if sc is not None else str(res)[:120]), TOPS[bid], len(ytr), len(teS)))
print("\nSUMMARY:", json.dumps(out_scores))

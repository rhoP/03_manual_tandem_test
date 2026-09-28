"""AMY manual-tandem recompute, geometry-gated: same generation-1 seed as
run_gen1.py's 'amy' run, but with a real bulb-height constraint column
(from the cluster geometry extraction, hull_sections.py/build_geometry_dataset.py
-- NOT modeled/predicted here) added to con_cols, so fsopt_tandem's own
propose_candidates() trains a DKL constraint surrogate on it and gates
proposals through trust.constraint_feasible (stay inside the observed
envelope of real bulb heights) exactly the way vera/manasa's con_cols
already work -- AMY's own config.py entry has con_cols=[] only because it
never had a bulb-height column to put there before this analysis.

Also cross-checks every proposed candidate against the large-N (n=1890)
ridge bulb-height model fit across the full AMY DB in the geometry
analysis (01_GP -- see the geo_lib.py / optimum_compare.py scratch work),
which is better-validated than the 884-point in-pipeline constraint
surrogate this script trains, and reports a hard-limit margin so the
"how sure are we" question has a number behind it, not just the gate flag.

    python run_gen2_amy_geo.py
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GP = HERE.parent / "01_GP"
DATA = HERE.parent / "data"
sys.path.insert(0, str(GP))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from dkl_mobo import config as dkl_config
from fsopt_tandem import config as tconfig, ingest, round as tround
from fsopt_tandem import state as tstate
from run_gen1 import make_gen1, GEN1_BATCHES, BATCH_SIZE

CASE = "amy"
OUT_SUBDIR = "amy_geo"
AMY_DIR = DATA / "Z489_MSC208_ECO-Retrofit_AMY04_Run03_OPT"
NPZ = {"NSGA1_000": AMY_DIR / "NSGA1_000_geometry.npz",
       "NSGA2_000": AMY_DIR / "NSGA2_000_geometry.npz"}
LO, HI = 8.0, 11.0                          # AMY's ct_ffw_HP01 HardLimits, from final.fso


def bulb_height_lookup() -> dict:
    """{Design ID: real bulb_height} for every AMY design the cluster geometry
    extraction covered (both NSGA batches, ~2700 designs) -- ground truth,
    not a model prediction."""
    out = {}
    for batch, path in NPZ.items():
        z = np.load(path)
        for did, bh in zip(z["design_id"], z["bulb_height"]):
            out[f"03_OPT/00/{batch}/{did}"] = float(bh)
    return out


def large_n_bulb_height_model():
    """The n=1890 ridge model from the geometry analysis (test R2=0.982,
    rmse=0.051), refit here on ALL matched valid AMY designs -- an
    independent, better-powered check on top of the in-pipeline constraint
    surrogate (which only sees the 884-design generation-1 seed)."""
    dfs = []
    for batch, path in NPZ.items():
        z = np.load(path)
        shp = pd.DataFrame(z["shape"], columns=list(z["shape_keys"]))
        shp["Design ID"] = [f"03_OPT/00/{batch}/{d}" for d in z["design_id"]]
        shp["bulb_height"] = z["bulb_height"]
        shp["ffw_valid"] = z["ffw_valid"]
        dfs.append(shp)
    geo = pd.concat(dfs, ignore_index=True)
    csv = pd.read_csv(dkl_config.CASES["amy"]["csv"])
    m = geo.merge(csv[["Design ID"]], on="Design ID", how="inner")
    m = m[m["ffw_valid"] == "1"]
    keys = [c for c in m.columns if c.startswith("_2_")]
    X, y = m[keys].values, m["bulb_height"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = StandardScaler().fit(Xtr)
    fit_tr = Ridge(alpha=3.0).fit(sc.transform(Xtr), ytr)
    r2 = r2_score(yte, fit_tr.predict(sc.transform(Xte)))
    rmse = np.sqrt(mean_squared_error(yte, fit_tr.predict(sc.transform(Xte))))
    sc_full = StandardScaler().fit(X)
    model_full = Ridge(alpha=3.0).fit(sc_full.transform(X), y)
    print(f"[large-N check] bulb-height ridge: n={len(y)}  test R2={r2:.3f}  rmse={rmse:.3f}  "
          f"(bound width {HI-LO:.1f})")
    return keys, sc_full, model_full, rmse


def main():
    bh = bulb_height_lookup()

    # gen-1 seed, exactly as run_gen1.py builds it, but with the real
    # bulb_height column attached and registered as a constraint.
    g = make_gen1(CASE)
    g["bulb_height"] = g[ingest.DESIGN_ID_COL].map(bh)
    missing = g["bulb_height"].isna().sum()
    if missing:
        print(f"gen1 seed: {len(g)} designs, {missing} without a matched bulb_height (dropped)")
    else:
        print(f"gen1 seed: {len(g)} designs, all matched to a real bulb_height")
    g = g.dropna(subset=["bulb_height"]).reset_index(drop=True)
    lo_obs, hi_obs = g["bulb_height"].min(), g["bulb_height"].max()
    print(f"  observed bulb_height range in this seed: [{lo_obs:.3f}, {hi_obs:.3f}]  "
          f"(true hard limit [{LO}, {HI}])")

    # register con_cols=['bulb_height'] for THIS run only (CASES is a plain
    # module-level dict -- mutated here, not in dkl_mobo/config.py, since
    # this is a one-off analysis run, not a change to the case's production
    # defaults).
    dkl_config.CASES["amy"] = {**dkl_config.CASES["amy"], "con_cols": ["bulb_height"]}

    out = HERE / OUT_SUBDIR
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    g.to_csv(out / "gen1_seed.csv", index=False)
    tconfig.TANDEM_OUT_BASE = HERE
    tconfig.register_case(f"manual_{OUT_SUBDIR}", base_dkl_case=CASE, out_subdir=OUT_SUBDIR)
    tstate.init_state(f"manual_{OUT_SUBDIR}")
    rep = tround.run_round(f"manual_{OUT_SUBDIR}", generation_df=g, batch_size=BATCH_SIZE, seed=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "objectives"}, indent=1, default=str))
    print(json.dumps(rep["objectives"], indent=1, default=str))

    cand = pd.read_csv(rep["candidates_csv"])
    print(f"\n{len(cand)} candidates proposed this round "
          f"({cand['trusted'].sum() if 'trusted' in cand else '?'} trusted)")
    if "bulb_height_pred" in cand.columns:
        print(cand[["bulb_height_pred", "bulb_height_sd"]].describe().round(3))

    # independent, large-N cross-check
    keys, sc_full, model_full, rmse = large_n_bulb_height_model()
    Xc = cand[keys].values
    pred = model_full.predict(sc_full.transform(Xc))
    cand["bulb_height_largeN_pred"] = pred
    for k_sigma in (0, 1, 2, 3):
        margin = k_sigma * rmse
        ok = (pred - margin >= LO) & (pred + margin <= HI)
        print(f"  k_sigma={k_sigma}: {ok.sum()}/{len(cand)} candidates predicted feasible "
              f"with a {margin:.3f} safety margin")
    cand["largeN_feasible_2sigma"] = (pred - 2 * rmse >= LO) & (pred + 2 * rmse <= HI)

    cand.to_csv(rep["candidates_csv"], index=False)
    cand.to_csv(Path(rep["candidates_csv"]).parent / "recompute_input.csv", index=False)
    print(f"\nwrote {rep['candidates_csv']}")


if __name__ == "__main__":
    main()

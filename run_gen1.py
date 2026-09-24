"""Manual fsopt_tandem run, step 1: train on generation 1 only, propose.

    python run_gen1.py amy tiger vera      # (or any subset)

Generation 1 = the first optimizer batch of each panel-flow DB
(03_OPT/00/NSGA1_000, plus SobolVicinity_000 for tiger), minus the
baseline-reference outlier rows listed in dkl_mobo.config.CASES.
Outputs land in ./<case>/ :
    gen1_seed.csv            the generation-1 designs fed to the surrogate
    round_0001/candidates.csv   surrogate-proposed designs (to recompute)
    round_0001/recompute_input.csv  same designs, shape params only
    state.json, history.csv ...  fsopt_tandem state (for the later `propose`)
"""
import json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GP = HERE.parent / "01_GP"
sys.path.insert(0, str(GP))

import pandas as pd
from dkl_mobo import config as dkl_config
from fsopt_tandem import config as tconfig, ingest, round as tround
from fsopt_tandem import state as tstate

GEN1_BATCHES = {"amy": ["NSGA1_000"], "vera": ["NSGA1_000"],
                "tiger": ["SobolVicinity_000", "NSGA1_000"]}
BATCH_SIZE = 20


def make_gen1(case, extra_cols=()):
    dkl_config.use_case(case)
    cfg = dkl_config.CASES[case]
    df = pd.read_csv(cfg["csv"])
    batch = df["Design ID"].str.rsplit("/", n=1).str[0].str.split("/").str[-1]
    m = batch.isin(GEN1_BATCHES[case]) & ~df["Design ID"].isin(cfg.get("exclude_design_ids", []))
    g = df[m]
    cols = ([c for c in g.columns if c.startswith("_")] + list(cfg["obj_cols"]) + list(extra_cols)
            + [c for c in cfg["con_cols"] if c in g.columns])
    cols = list(dict.fromkeys(cols))
    g = g[cols + ["Design ID"]].copy()
    g = g.rename(columns={"Design ID": ingest.DESIGN_ID_COL})
    g[ingest.GENERATION_COL] = 1
    # 'f(x)' is a positional marker in the source DB, not a shape column
    g = g.drop(columns=[c for c in g.columns if c == "f(x)"])
    return g


_DIAG = {}
_orig_propose = tround.propose_candidates
def _capture(*a, **k):
    r = _orig_propose(*a, **k)
    _DIAG.update(r[-1]); _DIAG["cand_x"] = r[0]
    return r
tround.propose_candidates = _capture


def run(case):
    g = make_gen1(case)
    out = HERE / case
    shutil.rmtree(out, ignore_errors=True)   # always a clean generation-1 start
    out.mkdir()
    g.to_csv(out / "gen1_seed.csv", index=False)
    tconfig.TANDEM_OUT_BASE = HERE          # keep everything inside this folder
    tconfig.register_case(f"manual_{case}", base_dkl_case=case, out_subdir=case)
    tstate.init_state(f"manual_{case}")
    rep = tround.run_round(f"manual_{case}", generation_df=g, batch_size=BATCH_SIZE, seed=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "objectives"}, indent=1, default=str))
    print(json.dumps(rep["objectives"], indent=1, default=str))
    cand = pd.read_csv(rep["candidates_csv"])
    # ALL candidates (not only the gated batch) + per-gate diagnostics, so the
    # human can choose what to recompute. Gates: nd=non-dominated, edge_ok,
    # calibrated=mu within k*sd of observed range, in_dist=kNN manifold guard.
    for g in ("nd", "edge_ok", "calibrated", "in_distribution"):
        if g in _DIAG and len(_DIAG[g]) == len(cand):
            cand["gate_" + g] = _DIAG[g]
    if "manifold_dist" in _DIAG and len(_DIAG["manifold_dist"]) == len(cand):
        cand["manifold_dist"] = _DIAG["manifold_dist"]
        cand["manifold_thresh"] = _DIAG["guard_thresh"]
    cand.to_csv(rep["candidates_csv"], index=False)
    cand.to_csv(Path(rep["candidates_csv"]).parent / "recompute_input.csv", index=False)


if __name__ == "__main__":
    for c in sys.argv[1:] or ["amy", "tiger", "vera"]:
        print(f"\n######## {c}")
        run(c)

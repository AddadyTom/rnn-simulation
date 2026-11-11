"""Easy-task plan, step 2, E4 (paper Fig 3e-f): on false-alarm trials, insula activity
looks like a mix of the light pattern (CR trials) and the waiting-for-water pattern (hits).

Pipeline as in the paper's Methods ("Dimensionality reduction and clustering"):
0.5 s bins, z-score each cell, two-stage Laplacian eigenmap (20 then 6 dimensions),
8 clusters, every time bin labelled by its cluster. Then
  * light cluster  = the cluster most over-represented on CR vs hit trials in the light
  * reward cluster = the cluster most over-represented on hit vs CR trials after the odor
  * trial-wise cosine similarity of cluster-presence vectors, within and between outcomes.
Pass: model FA trials contain both the light cluster and the reward cluster, as in the data.
Run on the real OI151 cells and on the fitted model plus each cell's own trial noise.
Writes runs/easy_e4.json."""
import json, sys
from pathlib import Path
import numpy as np, torch
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rnn_workbench.fit as F
from rnn_workbench.data import load_session
from rnn_workbench.train import load_model
from rnn_workbench.vagal_task import VagalOptoTask

run_dir = ROOT / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "fit_oi151")
cfg = F.load_fit_config(json.load(open(run_dir / "train_config.json"))["fit"])
cfg.session_dir = str(ROOT / cfg.session_dir)
sess = load_session(Path(cfg.session_dir), cfg.events_file, cfg.iscell_file, cfg.signal, cfg.fs, cfg.smooth_frames)
sess = F.select_cells(sess, cfg.n_fit_cells)  # the subset the run was fitted to (no-op for all cells)
task = VagalOptoTask(cfg.task)
torch.manual_seed(0)
data = F.build_trials(sess, task, np.random.default_rng(0), False)
x, steps, fs = data["x"], data["frame_steps"], cfg.fs
light = np.array([e["freq_hz"] > 0 for e in data["events"]])
licked = data["licked"].numpy().astype(bool)
outcome = np.where(light, np.where(licked, "FA", "CR"), np.where(licked, "hit", "miss"))
act = data["act"].numpy()
BIN = 7  # frames per 0.5 s bin, as in the paper
nb = act.shape[1] // BIN
LIGHT_BINS, AFTER_ODOR_BINS = range(2, 10), range(10, 18)  # 1-5 s, 5-9 s


def clusters(A, seed=0):
    B = A[:, : nb * BIN].reshape(A.shape[0], nb, BIN, -1).mean(2)  # trials, bins, cells
    X = B.reshape(-1, B.shape[2])
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    Z = SpectralEmbedding(20, affinity="nearest_neighbors", random_state=seed).fit_transform(X)
    Z = SpectralEmbedding(6, affinity="nearest_neighbors", random_state=seed).fit_transform(Z)
    return KMeans(8, n_init=10, random_state=seed).fit_predict(Z).reshape(B.shape[0], nb)


def analyse(lab):
    def frac(mask, bins, c):
        return (lab[mask][:, list(bins)] == c).any(1).mean()
    cr, fa, hit = outcome == "CR", outcome == "FA", outcome == "hit"
    light_c = max(range(8), key=lambda c: frac(cr, LIGHT_BINS, c) - frac(hit, LIGHT_BINS, c))
    reward_c = max(range(8), key=lambda c: frac(hit, AFTER_ODOR_BINS, c) - frac(cr, AFTER_ODOR_BINS, c))
    out = {"light_cluster_in": {}, "reward_cluster_in": {}}
    for o, m in (("CR", cr), ("FA", fa), ("hit", hit)):
        out["light_cluster_in"][o] = round(float(frac(m, LIGHT_BINS, light_c)), 2)
        out["reward_cluster_in"][o] = round(float(frac(m, AFTER_ODOR_BINS, reward_c)), 2)
    # trial-wise similarity of cluster presence (baseline bins dropped)
    V = np.stack([(lab[:, 2:] == c) for c in range(8)], -1).reshape(len(lab), -1).astype(float)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    S = V @ V.T
    def sim(a, b):
        M = S[np.ix_(a, b)]
        if a is b:
            return M[~np.eye(len(M), dtype=bool)].mean()
        return M.mean()
    idx = {o: np.flatnonzero(outcome == o) for o in ("CR", "FA", "hit")}
    out["similarity"] = {f"{a}-{b}": round(float(sim(idx[a], idx[b] if a != b else idx[a])), 3)
                         for a, b in (("FA", "CR"), ("FA", "hit"), ("CR", "hit"), ("CR", "CR"), ("hit", "hit"))}
    return out


cond = light * 2 + licked
resid = act.copy()
for c in np.unique(cond):
    resid[cond == c] -= act[cond == c].mean(0)

res = {"n_trials": {o: int((outcome == o).sum()) for o in ("CR", "FA", "hit", "miss")},
       "data": analyse(clusters(act))}
print("data", res["data"], flush=True)
for sd in sorted(run_dir.glob("seed_*")):
    ck = torch.load(sd / "model.pt", weights_only=False)
    m = load_model(sd)
    ro = F.Readout(act.shape[2], cfg.calcium_tau_ms or 1000, task.cfg.dt, False)
    ro.load_state_dict(ck["readout"], strict=False)
    with torch.no_grad():
        _, rates = m(x, noise=True)
        pred = ro(rates[:, :, torch.tensor(ck["unit_of_cell"])], steps).transpose(0, 1).numpy()
    rng = np.random.default_rng(0)
    noisy = pred + resid[rng.integers(len(resid), size=len(resid))]
    res[f"model {sd.name} + cells' trial noise"] = analyse(clusters(noisy))
    print(sd.name, res[f"model {sd.name} + cells' trial noise"], flush=True)
(ROOT / "runs" / ("easy_e4.json" if run_dir.name == "fit_oi151" else f"easy_e4_{run_dir.name}.json")).write_text(json.dumps(res, indent=1))

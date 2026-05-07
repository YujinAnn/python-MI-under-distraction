"""
Cross-condition EEGNet evaluation on pp1.mat with multiple test sets.

Training:
    label 1 <- {"S 01", "S 11"}     # left  MI, no distraction
    label 2 <- {"S 02", "S 12"}     # right MI, no distraction
    -> 80% of these trials are used for training,
       20% are held out as test_acc 1.

Additional held-out test sets (all trials of the listed condition):
    test_acc 2 : eyes-closed     {1: ["S 21"], 2: ["S 22"]}
    test_acc 3 : news            {1: ["S 31"], 2: ["S 32"]}
    test_acc 4 : numbers         {1: ["S 41"], 2: ["S 42"]}
    test_acc 5 : flicker         {1: ["S 51"], 2: ["S 52"]}
    test_acc 6 : stimulation     {1: ["S 61"], 2: ["S 62"]}

All test sets share the train-set z-score normalisation.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from reference_codes.read_main_experiment import DATA_DIR
from reference_codes.helper import load_data
from models.EEGNet_ver16 import EEG_Net


# ----------------------------- Hyperparameters ------------------------------
TRAIN_LABEL_MAP = {
    1: ["S 01", "S 11"],   # left-hand MI (calibration + clean feedback)
    2: ["S 02", "S 12"],   # right-hand MI (calibration + clean feedback)
}

# (test name, label_map) pairs for held-out evaluation conditions.
EVAL_LABEL_MAPS = [
    ("eyes-closed", {1: ["S 21"], 2: ["S 22"]}),
    ("news",        {1: ["S 31"], 2: ["S 32"]}),
    ("numbers",     {1: ["S 41"], 2: ["S 42"]}),
    ("flicker",     {1: ["S 51"], 2: ["S 52"]}),
    ("stimulation", {1: ["S 61"], 2: ["S 62"]}),
]

# "pp4.mat" works well. The accuracy different depending on subjects.
SUBJECT    = "pp1.mat"   # participant file in DATA_DIR (e.g. "pp1.mat", "pp2.mat")

IVAL_MS    = (0, 4500)
BAND_HZ    = (1, 50)
TARGET_FS  = 250

TEST_FRAC  = 0.2     # fraction of clean trials reserved for test_acc 1

BATCH_SIZE = 32
N_EPOCHS   = 60
LR         = 1e-3
SEED       = 0
# ----------------------------------------------------------------------------


def pick_device():
    """Return torch.device('cuda') when a GPU is available, else CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Device            : cuda ({torch.cuda.get_device_name(0)})")
    else:
        dev = torch.device("cpu")
        print(f"Device            : cpu")
    return dev


def _accuracy(model, X, y):
    with torch.no_grad():
        pred = model(X).argmax(1)
        return (pred == y).float().mean().item()


def train_and_evaluate(Y_tr, u_tr, eval_sets, device=None, seed=SEED):
    """
    Y_tr     : (N, 1, C, T) float32 -- training inputs
    u_tr     : (N,) int64           -- training labels
    eval_sets: list of (name, Y_te, u_te); each evaluated on its own.
    """
    if device is None:
        device = pick_device()

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Train-defined class ordering; remap user labels to 0-indexed for CE
    classes = sorted(np.unique(u_tr).tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    def remap(u): return np.array([label_to_idx[v] for v in u], dtype=np.int64)

    y_tr = remap(u_tr)

    # z-score using training stats only
    mean = Y_tr.mean(axis=(0, 3), keepdims=True)
    std  = Y_tr.std(axis=(0, 3), keepdims=True) + 1e-6

    def normalize(X): return ((X - mean) / std).astype(np.float32)

    X_tr_t = torch.from_numpy(normalize(Y_tr)).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)

    test_tensors = []
    for name, Y_te, u_te in eval_sets:
        X_te_t = torch.from_numpy(normalize(Y_te)).to(device)
        y_te_t = torch.from_numpy(remap(u_te)).to(device)
        test_tensors.append((name, X_te_t, y_te_t))

    n_channels = Y_tr.shape[2]
    n_samples  = Y_tr.shape[3]
    model = EEG_Net(nb_classes=len(classes),
                    nb_channels=n_channels,
                    nb_times=n_samples).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    n_train = X_tr_t.shape[0]
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        perm = torch.from_numpy(rng.permutation(n_train)).long()
        running = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        train_loss = running / n_train

        if epoch == 1 or epoch % 10 == 0 or epoch == N_EPOCHS:
            model.eval()
            tr_acc = _accuracy(model, X_tr_t, y_tr_t)
            test_accs = [(name, _accuracy(model, X, y)) for name, X, y in test_tensors]
            line = f"epoch {epoch:3d}  loss {train_loss:.4f}  train {tr_acc:.3f}"
            for k, (name, acc) in enumerate(test_accs, start=1):
                line += f"  test{k}({name}) {acc:.3f}"
            print(line)

    return model


if __name__ == "__main__":
    common = dict(ival_ms=IVAL_MS, band_hz=BAND_HZ, target_fs=TARGET_FS)

    subject_path = DATA_DIR / SUBJECT

    # 1) Clean condition: split 80/20 to get train + test_acc 1
    Y_clean, u_clean, info = load_data(subject_path,
                                       label_map=TRAIN_LABEL_MAP, **common)
    Y_tr, Y_te1, u_tr, u_te1 = train_test_split(
        Y_clean, u_clean, test_size=TEST_FRAC,
        stratify=u_clean, random_state=SEED,
    )

    # 2) The five distraction conditions: load each as its own test set
    eval_sets = [("clean-held", Y_te1, u_te1)]
    for name, lmap in EVAL_LABEL_MAPS:
        Y_te, u_te, _ = load_data(subject_path, label_map=lmap, **common)
        eval_sets.append((name, Y_te, u_te))

    def class_counts(u):
        uniq, cnt = np.unique(u, return_counts=True)
        return dict(zip(uniq.tolist(), cnt.tolist()))

    print(f"Subject           : {SUBJECT}")
    print(f"Train map         : {TRAIN_LABEL_MAP}")
    print(f"Train Y shape     : {Y_tr.shape}")
    print(f"Train class cnts  : {class_counts(u_tr)}")
    print(f"Effective fs      : {info['fs']} Hz")
    print()
    for k, (name, Y_te, u_te) in enumerate(eval_sets, start=1):
        print(f"test_acc {k} ({name:<12}) shape {Y_te.shape}  cnts {class_counts(u_te)}")

    device = pick_device()
    print()

    train_and_evaluate(Y_tr, u_tr, eval_sets, device=device)

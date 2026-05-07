"""
Read raw EEG (cnt_orig) and markers (mrk_orig) for the main experiment
from data/pp1.mat.

See data/Data_desciption.txt. Both cnt_orig and mrk_orig are 1x2 cell arrays:
    [0] = calibration phase (72 trials, no secondary task)
    [1] = feedback phase    (432 trials across 6 concatenated runs)

The .mat files are MATLAB v7.3 (HDF5), so we read them with h5py rather than
scipy.io.loadmat. Two v7.3 quirks handled below:
    - numeric arrays come out with axes reversed vs MATLAB (we transpose so
      x -> [samples, channels] and y -> [classes, trials])
    - MATLAB strings are uint16 UTF-16, cell arrays are arrays of HDF5 refs
"""

from pathlib import Path
import numpy as np
import h5py


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _decode(char_arr):
    return np.asarray(char_arr).ravel().astype(np.uint16).tobytes().decode("utf-16-le", errors="replace")


def _cell_of_strings(f, dset):
    return [_decode(f[r][()]) for r in np.asarray(dset).ravel()]


def _load_cnt(f, group):
    return {
        "x":     group["x"][()].T,                                      # [samples, channels]
        "clab":  _cell_of_strings(f, group["clab"]),
        "fs":    float(np.squeeze(group["fs"][()])),
        "yUnit": _decode(group["yUnit"][()]),
        "T":     np.atleast_1d(np.squeeze(group["T"][()])).astype(np.int64),
    }


def _load_mrk(f, group):
    ev = group["event"]
    return {
        "y":         group["y"][()].T,                                  # [classes, trials]
        "time":      np.atleast_1d(np.squeeze(group["time"][()])).astype(float),
        "className": _cell_of_strings(f, group["className"]),
        "event": {
            "desc": np.atleast_1d(np.squeeze(ev["desc"][()])),
            "type": _cell_of_strings(f, ev["type"]),
        },
    }


def load_main_experiment(mat_path):
    """Load cnt_orig and mrk_orig from a participant .mat file.

    Returns a dict with keys 'calibration' and 'feedback'. Each value is a
    dict {'cnt': <cnt_orig struct>, 'mrk': <mrk_orig struct>}.
    """
    with h5py.File(mat_path, "r") as f:
        cnt_refs = np.asarray(f["cnt_orig"]).ravel()
        mrk_refs = np.asarray(f["mrk_orig"]).ravel()
        phases = {}
        for name, i in (("calibration", 0), ("feedback", 1)):
            phases[name] = {
                "cnt": _load_cnt(f, f[cnt_refs[i]]),
                "mrk": _load_mrk(f, f[mrk_refs[i]]),
            }
    return phases


def rename_calibration_classes(data, renames=(("S 11", "S 01"), ("S 12", "S 02"))):
    """Rename class labels in the calibration phase only.

    Per Data_desciption.txt, calibration-phase left/right MI should be labeled
    '1'/'2' (not '11'/'12' — those are reserved for the 'clean' feedback
    condition). This relabels className and event.type to match.
    """
    mrk = data["calibration"]["mrk"]
    mapping = dict(renames)
    mrk["className"] = [mapping.get(c, c) for c in mrk["className"]]
    mrk["event"]["type"] = [mapping.get(t, t) for t in mrk["event"]["type"]]
    return data


def describe(phase_name, phase):
    cnt, mrk = phase["cnt"], phase["mrk"]
    duration_s = cnt["x"].shape[0] / cnt["fs"]
    uniq, counts = np.unique(mrk["event"]["desc"], return_counts=True)
    marker_hist = dict(zip(uniq.astype(int).tolist(), counts.tolist()))

    print(f"\n=== {phase_name} ===")
    print(f"  EEG x shape        : {cnt['x'].shape}   # [samples, channels]")
    print(f"  Duration           : {duration_s:.1f} s ({duration_s / 60:.1f} min)")
    print(f"  Sampling rate      : {cnt['fs']} Hz")
    print(f"  # channels         : {len(cnt['clab'])}")
    print(f"  yUnit              : {cnt['yUnit']!r}")
    print(f"  T (per file)       : {cnt['T'].tolist()}")
    print(f"  mrk y shape        : {mrk['y'].shape}   # [classes, trials]")
    print(f"  className          : {mrk['className']}")
    print(f"  # events           : {len(mrk['time'])}")
    print(f"  marker histogram   : {marker_hist}")


if __name__ == "__main__":
    data = load_main_experiment(DATA_DIR / "pp1.mat")

    # Relabel calibration left/right from 'S 11'/'S 12' to 'S 01'/'S 02'
    # so they are distinct from the 'clean' condition labels in feedback.
    rename_calibration_classes(data)

    describe("calibration", data["calibration"])
    describe("feedback",    data["feedback"])

    # Example: extract one left-hand motor-imagery trial from the feedback
    # "clean" condition (marker 11, trial onset). Cut from the event sample
    # to event + 4.5 s (trial length per the description).
    mrk = data["feedback"]["mrk"]
    cnt = data["feedback"]["cnt"]
    fs = cnt["fs"]
    left_clean_onsets_ms = mrk["time"][mrk["event"]["desc"] == 11]
    if len(left_clean_onsets_ms):
        t0 = int(left_clean_onsets_ms[0] / 1000.0 * fs)
        trial = cnt["x"][t0 : t0 + int(4.5 * fs), :]
        print(f"\nExample left-hand 'clean' trial (marker 11): shape {trial.shape}")

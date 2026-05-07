"""
Helper functions for loading and epoching EEG trials from the BCI-under-
distraction dataset. See train.py for the user-facing
hyperparameters (LABEL_MAP, IVAL_MS, BAND_HZ, TARGET_FS) that are passed
into load_data.
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

from .read_main_experiment import load_main_experiment, rename_calibration_classes


def _bandpass(x, fs, band, order=3):
    """Zero-phase Butterworth bandpass along axis 0.

    Processes one channel at a time and writes into a float32 output buffer
    to keep peak memory bounded on large continuous recordings (scipy's
    filtfilt keeps several internal float64 copies).
    """
    b, a = butter(order, np.asarray(band) / (fs / 2.0), btype="bandpass")
    out = np.empty(x.shape, dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = filtfilt(b, a, x[:, c]).astype(np.float32, copy=False)
    return out


def _epoch_continuous(x, fs, onsets_ms, ival_ms):
    """Cut epochs of shape (N_trials, T, C) from continuous EEG."""
    onsets_samp  = np.round(np.asarray(onsets_ms) * fs / 1000.0).astype(int)
    start_offset = int(round(ival_ms[0] * fs / 1000.0))
    n_samples    = int(round((ival_ms[1] - ival_ms[0]) * fs / 1000.0))
    n_total      = x.shape[0]
    out = []
    for t in onsets_samp:
        s = t + start_offset
        e = s + n_samples
        if 0 <= s and e <= n_total:
            out.append(x[s:e, :])
    return np.stack(out, axis=0) if out else np.zeros((0, n_samples, x.shape[1]), dtype=x.dtype)


def load_data(mat_path, label_map, ival_ms, band_hz, target_fs):
    """Load and epoch EEG trials from a participant .mat file.

    Parameters
    ----------
    mat_path  : path to the participant .mat file
    label_map : dict[int, list[str]] mapping integer labels -> className strings
    ival_ms   : (start_ms, end_ms) epoch window relative to trial onset
    band_hz   : (low_hz, high_hz) bandpass filter
    target_fs : effective sampling rate (Hz) after optional downsampling.
                If >= original fs, no downsampling is performed.

    Returns
    -------
    Y    : np.ndarray, shape (N, 1, C, T), float32  -- input tensor
    u    : np.ndarray, shape (N,),         int64    -- integer labels
    info : dict with 'fs' (effective Fs), 'channels', 'label_map'
    """
    data = load_main_experiment(mat_path)
    rename_calibration_classes(data)

    epoch_blocks, label_blocks = [], []
    channels = None

    for phase_name in ("calibration", "feedback"):
        phase = data[phase_name]
        cnt, mrk = phase["cnt"], phase["mrk"]
        fs = cnt["fs"]
        channels = cnt["clab"]

        x_filt = _bandpass(cnt["x"], fs, band_hz)
        class_names = mrk["className"]        # e.g. ['n/d', 'S  5', 'S 01', 'S 02']
        y_mat       = mrk["y"]                # (n_classes, n_events), one-hot per event
        event_times = mrk["time"]

        # event.type is just 'Stimulus' / 'New Segment', so select trials by
        # combining className + the one-hot y matrix.
        for label, wanted_names in label_map.items():
            mask = np.zeros(y_mat.shape[1], dtype=bool)
            for name in wanted_names:
                if name in class_names:
                    mask |= (y_mat[class_names.index(name)] == 1)
            if not mask.any():
                continue
            epochs = _epoch_continuous(x_filt, fs, event_times[mask], ival_ms)
            if epochs.size == 0:
                continue
            epoch_blocks.append(epochs)
            label_blocks.append(np.full(epochs.shape[0], label, dtype=np.int64))

    if not epoch_blocks:
        raise RuntimeError("No epochs matched label_map. Check class names.")

    X = np.concatenate(epoch_blocks, axis=0)   # (N, T, C) at original fs
    u = np.concatenate(label_blocks, axis=0)

    # Downsample along T
    if target_fs and target_fs < fs:
        X = resample_poly(X, up=int(target_fs), down=int(fs), axis=1).astype(np.float32)
        eff_fs = target_fs
    else:
        eff_fs = fs

    X = X.transpose(0, 2, 1)                         # (N, C, T)
    Y = X[:, np.newaxis, :, :].astype(np.float32)    # (N, 1, C, T)

    return Y, u, {"fs": eff_fs, "channels": channels, "label_map": label_map}

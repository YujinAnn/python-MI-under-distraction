# BCI Under Distraction — EEGNet Cross-Condition Evaluation

Train an EEGNet motor-imagery classifier on the "clean" condition of the
BCI-under-distraction dataset and evaluate it on five held-out distraction
conditions (eyes-closed, news, numbers, flicker, electrical stimulation).

## Repository layout

```
.
├── train.py                      # entry point: train + evaluate
├── models/
│   └── EEGNet_ver16.py           # EEGNet model (PyTorch)
├── reference_codes/
│   ├── helper.py                 # epoching, bandpass, downsampling
│   └── read_main_experiment.py   # .mat (HDF5 v7.3) loader
├── requirements.txt
└── data/                         # put pp1.mat here (not committed)
```

Run `train.py` from the repo root so that `models.` and `reference_codes.`
package imports resolve.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Data

The `.mat` files are not in the repository. Download the dataset
*BCI under distraction: Motor imagery in a pseudo realistic environment*
from TU Berlin's DepositOnce:

https://depositonce.tu-berlin.de/items/0f01eb46-4e6e-427a-9a68-b264a839615f

The paper is "Motor Imagery Under Distraction— An Open Access BCI Dataset"
https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.566147/full

Place participant files into the `data/` folder at the repo root:

```
data/
└── pp1.mat
```

## Run

```bash
python train.py
```

The script auto-detects CUDA (falls back to CPU) and prints train accuracy
and accuracy on each of the six test sets every 10 epochs.

To train on a different participant, change `SUBJECT` at the top of
`train.py`:

```python
SUBJECT = "pp2.mat"   # any .mat file inside data/
```

### Training labels

| Label | Markers         | Condition                        |
|-------|-----------------|----------------------------------|
| 1     | `S 01`, `S 11`  | left-hand MI (calibration + clean feedback)  |
| 2     | `S 02`, `S 12`  | right-hand MI (calibration + clean feedback) |

80% of these trials are used for training; 20% become `test_acc 1`
("clean-held"). The remaining held-out evaluation conditions:

| test_acc | Condition      | Markers         |
|----------|----------------|-----------------|
| 2        | eyes-closed    | `S 21`, `S 22`  |
| 3        | news           | `S 31`, `S 32`  |
| 4        | numbers        | `S 41`, `S 42`  |
| 5        | flicker        | `S 51`, `S 52`  |
| 6        | stimulation    | `S 61`, `S 62`  |

All test sets share the train-set z-score normalization.

## Hyperparameters

Edit the constants at the top of `train.py`:

- `SUBJECT = "pp1.mat"` — participant file inside `data/`
- `IVAL_MS = (0, 4500)` — epoch window relative to trial onset (ms)
- `BAND_HZ = (1, 50)` — bandpass filter (Hz)
- `TARGET_FS = 250` — effective sampling rate after downsampling (Hz)
- `TEST_FRAC = 0.2` — fraction of clean trials held out as `test_acc 1`
- `BATCH_SIZE`, `N_EPOCHS`, `LR`, `SEED`

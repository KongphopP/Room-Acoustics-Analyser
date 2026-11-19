"""Load the Arni dataset (Aalto, Zenodo 6985104, CC BY 4.0) into DuckDB.

What it does:
1. combinations_setup.csv -> table arni_setups (1 row per panel setup)
2. WAV file names         -> table rirs (1 row per file, test split)

It reads only WAV headers (fast). The audio itself is read later with
read_rir_16k(), which always returns 16 kHz, so every step after this one
sees the same sample rate (decision 2026-11-19).

Usage (from the repo root, after `python scripts/init_db.py --fresh`):
    python scripts/load_arni.py
"""
import re
from math import ceil
from pathlib import Path

import duckdb
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "acoustics.duckdb"
RAW = ROOT / "data" / "raw"
ARNI_DIR = RAW / "arni" / "ir_0-5"
ARNI_CSV = RAW / "arni" / "combinations_setup.csv"

FS = 16000                          # project sample rate
ROOM_DIMS = (8.9, 6.3, 3.6)         # Arni room L x W x H in metres
N_PANELS = 55
NAME_RE = re.compile(r"IR_numClosed_(\d+)_numComb_(\d+)_mic_(\d+)_sweep_(\d+)\.wav$")


def read_setups(csv_path=ARNI_CSV):
    """Read combinations_setup.csv into a DataFrame (num_comb, num_closed, panels_open).

    The file is not a standard CSV (';' separator, 2-line header, mixed line
    endings), so we parse it with plain Python. CSV value 1 = panel OPEN.
    """
    lines = Path(csv_path).read_text().splitlines()[2:]   # skip 2-line header
    rows = []
    for line in lines:
        values = [v for v in line.split(";") if v.strip() != ""]
        if not values:
            continue
        num_comb = int(float(values[0]))
        panels = [int(float(v)) for v in values[1:]]
        if len(panels) != N_PANELS:
            raise ValueError(f"numComb {num_comb}: expected {N_PANELS} panels, got {len(panels)}")
        rows.append({
            "num_comb": num_comb,
            "num_closed": N_PANELS - sum(panels),
            "panels_open": "".join(str(p) for p in panels),
        })
    return pd.DataFrame(rows)


def n_samples_16k(n_original, fs_original):
    """Length after resampling to 16 kHz (same rule resample_poly uses)."""
    return ceil(n_original * FS / fs_original)


def scan_wavs(folder=ARNI_DIR):
    """Read file names + WAV headers into a DataFrame shaped like the rirs table."""
    rows = []
    for path in sorted(Path(folder).glob("*.wav")):
        m = NAME_RE.search(path.name)
        if m is None:
            raise ValueError(f"Unexpected file name: {path.name}")
        num_closed, num_comb, mic, sweep = map(int, m.groups())
        info = sf.info(path)                       # header only, no audio
        rows.append({
            "rir_uid": f"arni_c{num_comb}_m{mic}_s{sweep}",
            "dataset": "arni",
            "split": "test",
            "source_file": str(path.relative_to(RAW)),
            "channel": 0,
            "fs_original": info.samplerate,
            "fs": FS,
            "n_samples": n_samples_16k(info.frames, info.samplerate),
            "room_type": "shoebox",
            "room_l": ROOM_DIMS[0], "room_w": ROOM_DIMS[1], "room_h": ROOM_DIMS[2],
            "arni_num_comb": num_comb,
            "_num_closed": num_closed,             # only used for a check below
        })
    return pd.DataFrame(rows)


def read_rir_16k(source_file, channel=0):
    """Read one RIR from data/raw and return it at 16 kHz (mono numpy array)."""
    x, fs = sf.read(RAW / source_file, always_2d=True)
    x = x[:, channel]
    if fs != FS:
        x = resample_poly(x, FS, fs)               # polyphase resampling with anti-alias filter
    return x


def load_arni(con):
    """Insert Arni setups and RIRs into an open DuckDB connection."""
    setups = read_setups()
    rirs = scan_wavs()

    # Check: numClosed in the file name must match the CSV row (proven 19 Nov)
    merged = rirs.merge(setups, left_on="arni_num_comb", right_on="num_comb")
    bad = merged[merged["_num_closed"] != merged["num_closed"]]
    if len(merged) != len(rirs) or len(bad) > 0:
        raise ValueError(f"{len(rirs) - len(merged)} files without a setup, "
                         f"{len(bad)} files with wrong numClosed")
    rirs = rirs.drop(columns="_num_closed")

    # Give each DataFrame a name that cannot clash with a real table name
    con.register("setups_df", setups)
    con.register("rirs_df", rirs)
    con.execute("INSERT INTO arni_setups SELECT * FROM setups_df")
    con.execute(f"INSERT INTO rirs ({', '.join(rirs.columns)}) SELECT * FROM rirs_df")
    return len(setups), len(rirs)


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    n_setups, n_rirs = load_arni(con)
    print(f"Loaded {n_setups} setups and {n_rirs} RIRs")
    print(con.sql("""
        SELECT s.num_closed, count(DISTINCT s.num_comb) AS setups, count(*) AS rirs
        FROM rirs r JOIN arni_setups s ON r.arni_num_comb = s.num_comb
        GROUP BY s.num_closed ORDER BY s.num_closed
    """))

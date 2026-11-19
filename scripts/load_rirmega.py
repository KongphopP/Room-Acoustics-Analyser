"""Download RIR-Mega (Hugging Face, mandipgoswami/rirmega) and load it into DuckDB.

What it does:
1. Downloads the two v1 subsets (4,000 WAVs, ~0.7 GB) into data/raw/rirmega/
   the first time it runs. Later runs reuse the local files.
2. metadata.csv -> table rirs (1 row per RIR, channel 0, train split)
3. metrics      -> table reference_labels (dataset labels, for EDA only)

Usage (from the repo root, after init_db.py and load_arni.py):
    python scripts/load_rirmega.py
"""
import ast
from math import ceil, dist
from pathlib import Path

import duckdb
import pandas as pd
import soundfile as sf
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "acoustics.duckdb"
RAW = ROOT / "data" / "raw"
RIRMEGA_DIR = RAW / "rirmega"

REPO = "mandipgoswami/rirmega"
SUBSETS = {                                  # short name -> folder in the repo
    "1k": "data/audio/rir_output_1k",        # linear 8-mic array
    "3k": "data/audio/rir_output_3k_circ",   # circular 8-mic array
}
FS = 16000
CHANNEL = 0                                  # we use mic 0 of the 8-mic array


def download_rirmega():
    """Download the two subsets into data/raw/rirmega/ (skips files already there)."""
    snapshot_download(
        REPO, repo_type="dataset", local_dir=RIRMEGA_DIR,
        allow_patterns=[f"{folder}/*" for folder in SUBSETS.values()],
    )


def parse(text):
    """Turn a Python-style text like "[1.0, 2.0]" or "{'a': 1}" into a Python object."""
    return ast.literal_eval(text)


def read_subset(short, folder):
    """Read one subset's metadata.csv and return (rirs rows, reference_labels rows)."""
    meta_path = RIRMEGA_DIR / folder / "metadata.csv"
    meta = duckdb.sql(f"SELECT * FROM '{meta_path}'").df()

    rirs, labels = [], []
    for row in meta.itertuples():
        wav_path = RIRMEGA_DIR / folder / row.wav
        info = sf.info(wav_path)                       # header only, no audio
        if info.samplerate != row.fs:
            raise ValueError(f"{row.wav}: header fs {info.samplerate} != metadata fs {row.fs}")

        room = parse(row.room_size)
        src = parse(row.source)
        # 'microphone' is the CENTRE of the array; channel 0 is array['mics'][0]
        mic = parse(row.array)["mics"][CHANNEL]
        uid = f"rirmega_{short}_{row.id}_ch{CHANNEL}"

        rirs.append({
            "rir_uid": uid,
            "dataset": "rirmega",
            "split": "train",                          # all simulated data = train
            "source_file": str(wav_path.relative_to(RAW)),
            "channel": CHANNEL,
            "fs_original": info.samplerate,
            "fs": FS,
            "n_samples": ceil(info.frames * FS / info.samplerate),
            "room_type": row.family,                   # e.g. 'office_small', 'classroom'
            "room_l": room[0], "room_w": room[1], "room_h": room[2],
            "src_x": src[0], "src_y": src[1], "src_z": src[2],
            "mic_x": mic[0], "mic_y": mic[1], "mic_z": mic[2],
            "distance_m": round(dist(src, mic), 3),
        })

        # Labels that came with the dataset. band_hz = 0 means broadband.
        m = parse(row.metrics)
        for metric in ("rt60", "drr_db", "c50_db", "c80_db"):
            name = "rt60_s" if metric == "rt60" else metric
            labels.append((uid, 0, "rirmega_metadata", name, m[metric]))
        for band, value in m["band_rt60s"].items():
            labels.append((uid, int(band), "rirmega_metadata", "rt60_s", value))

    labels = pd.DataFrame(labels, columns=["rir_uid", "band_hz", "source", "metric", "value"])
    return pd.DataFrame(rirs), labels


def load_rirmega(con):
    """Insert all RIR-Mega RIRs and their labels into an open DuckDB connection."""
    if not all((RIRMEGA_DIR / f / "metadata.csv").exists() for f in SUBSETS.values()):
        print("Downloading RIR-Mega (~0.7 GB, first run only)...")
        download_rirmega()

    parts = [read_subset(short, folder) for short, folder in SUBSETS.items()]
    rirs = pd.concat([p[0] for p in parts], ignore_index=True)
    labels = pd.concat([p[1] for p in parts], ignore_index=True)

    con.register("rirs_df", rirs)
    con.register("labels_df", labels)
    con.execute(f"INSERT INTO rirs ({', '.join(rirs.columns)}) SELECT * FROM rirs_df")
    con.execute("INSERT INTO reference_labels SELECT * FROM labels_df")
    return len(rirs), len(labels)


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    n_rirs, n_labels = load_rirmega(con)
    print(f"Loaded {n_rirs} RIRs and {n_labels} reference labels")
    print(con.sql("""
        SELECT dataset, split, count(*) AS rirs,
               round(min(distance_m), 2) AS min_dist, round(max(distance_m), 2) AS max_dist
        FROM rirs GROUP BY ALL ORDER BY ALL
    """))
    print(con.sql("""
        SELECT room_type, count(*) AS rirs FROM rirs
        WHERE dataset = 'rirmega' GROUP BY ALL ORDER BY rirs DESC
    """))

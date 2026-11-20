"""Run waveform data checks on every RIR and store the results in data_checks.

Checks (1 row per RIR per check):
- finite            every sample is a real number (no NaN / inf)
- not_empty         the peak is above zero (file is not silent)
- not_clipped       at most 3 samples sit at the peak level (a clipped
                    recording has a flat top: many samples at the same max)
- onset_after_direct_path
                    the loudest point (direct sound) does not arrive earlier
                    than physics allows: distance / speed of sound. Catches
                    wrong positions or a wrong channel.
- label_fits_file_length (RIR-Mega only)
                    the dataset's own RT60 label is possible for this file:
                    a decay of 35 dB at the label's rate (35/60 x max band
                    RT60) fits in the file after the onset. This checks the
                    LABEL, not the audio: on 19 Nov a 50-file probe showed the
                    band energy of every file falls 70-107 dB before the file
                    ends, so a fail means the Sabine label is too long.
                    Do not use it to drop RIRs from training.

Checks read the RAW file at its original sample rate on purpose:
resampling smooths a clipped flat top and would hide it.

Usage (from the repo root, after the loaders):
    python scripts/run_checks.py
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "acoustics.duckdb"
RAW = ROOT / "data" / "raw"

SPEED_OF_SOUND = 343.0      # m/s at about 20 degC
ONSET_TOLERANCE_MS = 1.0    # allow 1 ms for position and timing error
MAX_PEAK_SAMPLES = 3
CLIP_LEVEL = 0.999          # "at the peak" = within 0.1% of the peak value
T30_DROP_DB = 35            # T30 fits the decay from -5 to -35 dB


def check_rir(x, fs, distance_m, max_label_rt60=None):
    """Return a list of (check_name, passed, value, threshold) for one RIR."""
    results = []

    finite = bool(np.all(np.isfinite(x)))
    results.append(("finite", finite, None, None))
    if not finite:
        return results                      # other checks are meaningless

    peak = float(np.max(np.abs(x)))
    results.append(("not_empty", peak > 0, peak, 0.0))
    if peak == 0:
        return results

    n_at_peak = int(np.sum(np.abs(x) >= CLIP_LEVEL * peak))
    results.append(("not_clipped", n_at_peak <= MAX_PEAK_SAMPLES, n_at_peak, MAX_PEAK_SAMPLES))

    onset = int(np.argmax(np.abs(x)))
    onset_ms = onset / fs * 1000
    direct_ms = distance_m / SPEED_OF_SOUND * 1000
    margin_ms = onset_ms - direct_ms        # how much later than the direct path
    results.append(("onset_after_direct_path", margin_ms >= -ONSET_TOLERANCE_MS,
                    round(margin_ms, 3), -ONSET_TOLERANCE_MS))

    if max_label_rt60 is not None:
        after_onset_s = (len(x) - onset) / fs
        needed_s = T30_DROP_DB / 60 * max_label_rt60
        results.append(("label_fits_file_length", after_onset_s >= needed_s,
                        round(after_onset_s, 3), round(needed_s, 3)))
    return results


def run_checks(con):
    """Check every RIR in the rirs table and replace the data_checks rows."""
    rirs = con.execute("""
        SELECT r.rir_uid, r.source_file, r.channel, r.distance_m,
               max(l.value) AS max_label_rt60          -- NULL for Arni (no labels)
        FROM rirs r
        LEFT JOIN reference_labels l
               ON l.rir_uid = r.rir_uid AND l.metric = 'rt60_s' AND l.band_hz > 0
        GROUP BY ALL ORDER BY r.rir_uid
    """).fetchall()

    rows = []
    for i, (uid, src, ch, dist_m, max_rt60) in enumerate(rirs, start=1):
        x, fs = sf.read(RAW / src, always_2d=True)
        for name, passed, value, threshold in check_rir(x[:, ch], fs, dist_m, max_rt60):
            rows.append((uid, name, passed, value, threshold))
        if i % 1000 == 0:
            print(f"  checked {i} / {len(rirs)}")

    df = pd.DataFrame(rows, columns=["rir_uid", "check_name", "passed", "value", "threshold"])
    con.register("checks_df", df)
    con.execute("DELETE FROM data_checks")            # re-running replaces old results
    con.execute("""INSERT INTO data_checks (rir_uid, check_name, passed, value, threshold)
                   SELECT * FROM checks_df""")
    return len(rirs), len(df)


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    n_rirs, n_rows = run_checks(con)
    print(f"Checked {n_rirs} RIRs -> {n_rows} check rows\n")
    print(con.sql("""
        SELECT r.dataset, c.check_name,
               count(*) AS n, count(*) FILTER (WHERE NOT c.passed) AS failed,
               round(min(c.value), 2) AS min_value, round(max(c.value), 2) AS max_value
        FROM data_checks c JOIN rirs r USING (rir_uid)
        GROUP BY ALL ORDER BY ALL
    """))

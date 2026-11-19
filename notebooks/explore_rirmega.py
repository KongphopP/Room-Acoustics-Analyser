"""Explore the RIR-Mega dataset (Hugging Face).

What we found (19 Nov 2026):
- Use the v1 CSVs: rir_output_1k (1,000 rows) + rir_output_3k_circ (3,000 rows).
  metadata_v2.parquet has RT60 columns that are empty (NULL) in every row.
- Nested columns (metrics, absorption_bands, ...) are Python dict strings,
  not valid JSON, so we read them with ast.literal_eval.
- band_rt60s labels match the Sabine formula exactly; they are not measured
  from the waveform. Example: a warehouse RIR is only 1.27 s long but its
  Sabine label says 7.5 s. So we will measure our own labels (ISO 3382).
- WAV files: 16 kHz, 8 channels (one per mic in the array). We use channel 0.
"""
import ast
import duckdb
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "mandipgoswami/rirmega"
SUBSET = "data/audio/rir_output_1k"

# 1) Load the metadata and turn the 'metrics' text into a Python dict
meta = hf_hub_download(REPO, f"{SUBSET}/metadata.csv", repo_type="dataset")
rows = duckdb.sql(f"SELECT id, family, wav, metrics FROM '{meta}'").fetchall()
parsed = [(i, fam, wav, ast.literal_eval(m)) for i, fam, wav, m in rows]

# 2) Pick two RIRs: the first one, and the one with the longest band RT60
#    (the most reverberant room is the hardest case)
first = parsed[0]
worst = max(parsed, key=lambda r: max(r[3]["band_rt60s"].values()))

for rid, fam, wav, m in (first, worst):
    path = hf_hub_download(REPO, f"{SUBSET}/{wav}", repo_type="dataset")
    x, fs = sf.read(path)
    h = x if x.ndim == 1 else x[:, 0]          # keep channel 0 only

    # Broadband decay curve (no octave filtering yet, just checking depth)
    onset = int(np.argmax(np.abs(h)))          # loudest point = direct sound
    e = h[onset:] ** 2                         # energy from the onset onwards
    edc = np.cumsum(e[::-1])[::-1]             # add up energy from the end backwards
    edc_db = 10 * np.log10(edc / edc[0] + 1e-300)
    depth = edc_db[int(len(edc_db) * 0.95)]    # skip the last 5% (drops to -inf)

    max_band = max(m["band_rt60s"].values())
    print(f"\n{rid} | {fam} | {wav}")
    print(f"  fs={fs} shape={x.shape} length={len(h)/fs:.3f}s onset={onset/fs*1000:.1f}ms")
    print(f"  label rt60={m['rt60']:.3f}s  max band rt60 (Sabine)={max_band:.3f}s")
    print(f"  decay curve depth at 95% length = {depth:.1f} dB")
    # A curve needs about 35/60 x RT60 seconds to fall 35 dB (needed for T30)
    print(f"  time needed to reach -35 dB = {35/60*max_band:.3f}s after onset")
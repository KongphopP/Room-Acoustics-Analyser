"""Explore the Arni dataset (Aalto University, Zenodo 6985104, CC BY 4.0).

What we found (19 Nov 2026), using IR_Arni_upload_numClosed_0-5.zip:
- Room: rectangular, 8.9 x 6.3 x 3.6 m. 1 source, 5 mics (see Arni_layout.jpg).
- combinations_setup.csv is not a standard CSV (';' separator, 2-line quoted
  header, mixed CRLF/LF), so we read it with plain Python.
- File name numComb = row number in the CSV. CSV value 1 = panel OPEN.
  Proven on all 11,130 files: numClosed + (number of 1s in the row) = 55.
- 456 setups; 54 setups have 20 files instead of 25 (270 files missing).
- WAV: 44.1 kHz, mono, float, 2.4 s long.
- Octave bands 125 Hz - 4 kHz: peak-to-noise 66-87 dB (ISO 3382 T30 needs
  >= 45 dB). 63 Hz: only ~42 dB and T30 ~ 8 s = noise, not usable.
"""
import glob
import re
from collections import Counter

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

FOLDER = "data/raw/arni/ir_0-5"
CSV = "data/raw/arni/combinations_setup.csv"
BANDS = [63, 125, 250, 500, 1000, 2000, 4000]   # octave band centres (Hz)


def octave_filter(x, fs, fc):
    """Band-pass filter one octave wide around fc (6th-order Butterworth)."""
    lo, hi = fc / np.sqrt(2), fc * np.sqrt(2)
    sos = butter(3, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)                  # zero-phase: no time shift


def t30(h, fs):
    """T30 from the Schroeder decay curve (-5 to -35 dB). NaN if too shallow.
    Note: no noise-floor truncation yet, so noisy bands read too long."""
    e = h ** 2
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-300)
    if edc_db.min() > -35:
        return np.nan
    i0, i1 = np.argmax(edc_db <= -5), np.argmax(edc_db <= -35)
    slope = np.polyfit(np.arange(i0, i1) / fs, edc_db[i0:i1], 1)[0]
    return -60 / slope


# 1) Read the panel setup CSV (plain Python handles the mixed line endings)
with open(CSV) as fh:
    lines = fh.read().splitlines()[2:]          # skip the 2-line header
rows = [[float(v) for v in line.split(";") if v != ""]
        for line in lines if line.strip()]
ones = {int(r[0]): int(sum(r[1:])) for r in rows}
print(len(rows), "combinations,", len(rows[0]) - 1, "panels")

# 2) Match file names to CSV rows and check the rule numClosed + ones == 55
files = sorted(glob.glob(f"{FOLDER}/*.wav"))
pat = re.compile(r"numClosed_(\d+)_numComb_(\d+)_mic_(\d+)_sweep_(\d+)")
parsed = [tuple(map(int, pat.search(f).groups())) for f in files]
print("rule holds:", Counter(nc + ones[c] == 55 for nc, c, _, _ in parsed))

per_setup = Counter((nc, c) for nc, c, _, _ in parsed)
print("setups:", len(per_setup), "| files per setup:", Counter(per_setup.values()))

# 3) Per-band signal quality and T30 for the driest room and 5 closed panels
for nc in (0, 5):
    f = sorted(glob.glob(f"{FOLDER}/IR_numClosed_{nc}_*_mic_1_sweep_1.wav"))[0]
    x, fs = sf.read(f)
    onset = int(np.argmax(np.abs(x)))
    print(f"\n{f.split('/')[-1]}  (fs {fs}, onset {onset/fs*1000:.1f} ms)")
    print("  band   peak-to-noise   T30")
    for fc in BANDS:
        y = octave_filter(x, fs, fc)
        peak = 10 * np.log10(np.max(y[onset:onset + int(0.05 * fs)] ** 2))
        noise = 10 * np.log10(np.mean(y[-int(0.2 * fs):] ** 2))  # last 0.2 s
        print(f"  {fc:5d}  {peak - noise:6.1f} dB     {t30(y[onset:], fs):.3f} s")
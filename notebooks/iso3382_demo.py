import numpy as np

def schroeder_edc_db(h):
    """Turn an RIR into a smooth decay curve (in dB).
    Each point = how much energy is left from that moment to the end."""
    e = h ** 2                                     # energy = signal squared
    edc = np.cumsum(e[::-1])[::-1]                 # add up energy from the end backwards
    return 10 * np.log10(edc / edc[0] + 1e-300)   # start at 0 dB; tiny number avoids log(0)

def decay_time(edc_db, fs, start_db, end_db):
    """Fit a straight line between two dB levels on the decay curve,
    then stretch it to a 60 dB drop. Returns time in seconds."""
    i0 = np.argmax(edc_db <= start_db)             # first point at or below start_db
    i1 = np.argmax(edc_db <= end_db)               # first point at or below end_db
    t = np.arange(i0, i1) / fs                     # sample index -> seconds
    slope, _ = np.polyfit(t, edc_db[i0:i1], 1)     # line slope in dB per second (negative)
    return -60 / slope                             # time to drop 60 dB

def clarity_db(h, fs, ms):
    """C50 / C80: energy before `ms` milliseconds vs energy after it (in dB).
    Higher = clearer sound."""
    k = int(fs * ms / 1000)                        # milliseconds -> sample index
    e = h ** 2
    return 10 * np.log10(e[:k].sum() / e[k:].sum())

# Test RIR: random noise that fades out with a known RT60 of 0.5 s
fs, rt60_true = 16000, 0.5
t = np.arange(int(1.5 * fs)) / fs                  # 1.5 seconds of time points
rng = np.random.default_rng(0)                     # fixed seed = same result every run
h = rng.standard_normal(len(t)) * 10 ** (-3 * t / rt60_true)   # fades 60 dB in 0.5 s

edc = schroeder_edc_db(h)
print(f"EDT = {decay_time(edc, fs,  0, -10):.3f} s")   # early decay: 0 to -10 dB
print(f"T20 = {decay_time(edc, fs, -5, -25):.3f} s")   # -5 to -25 dB
print(f"T30 = {decay_time(edc, fs, -5, -35):.3f} s")   # -5 to -35 dB
print(f"C50 = {clarity_db(h, fs, 50):.2f} dB")         # speech clarity
print(f"C80 = {clarity_db(h, fs, 80):.2f} dB")         # music clarity
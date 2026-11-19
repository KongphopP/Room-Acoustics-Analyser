import ast
import duckdb
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "mandipgoswami/rirmega"
SUBSET = "data/audio/rir_output_1k"

# 1) อ่าน metadata แล้วแปลง metrics (Python dict string) ด้วย ast
meta = hf_hub_download(REPO, f"{SUBSET}/metadata.csv", repo_type="dataset")
rows = duckdb.sql(f"SELECT id, family, wav, metrics FROM '{meta}'").fetchall()
parsed = [(i, fam, wav, ast.literal_eval(m)) for i, fam, wav, m in rows]

# 2) เลือกไฟล์แรก + ไฟล์ที่ band RT60 สูงสุด (กรณีก้องที่สุด)
first = parsed[0]
worst = max(parsed, key=lambda r: max(r[3]["band_rt60s"].values()))

for rid, fam, wav, m in (first, worst):
    path = hf_hub_download(REPO, f"{SUBSET}/{wav}", repo_type="dataset")
    x, fs = sf.read(path)
    h = x if x.ndim == 1 else x[:, 0]          # ใช้ channel แรกถ้ามีหลาย channel

    # Broadband Schroeder EDC (ยังไม่กรอง band, แค่ดูความลึก)
    onset = int(np.argmax(np.abs(h)))
    e = h[onset:] ** 2
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-300)
    depth = edc_db[int(len(edc_db) * 0.95)]    # ไม่ใช้ 5% ท้ายสุด (ลดลงจนเป็น -inf)

    max_band = max(m["band_rt60s"].values())
    print(f"\n{rid} | {fam} | {wav}")
    print(f"  fs={fs} shape={x.shape} length={len(h)/fs:.3f}s onset={onset/fs*1000:.1f}ms")
    print(f"  label rt60={m['rt60']:.3f}s  max band rt60 (Sabine)={max_band:.3f}s")
    print(f"  EDC depth at 95% length = {depth:.1f} dB")
    print(f"  time needed to reach -35 dB ≈ {35/60*max_band:.3f}s after onset")
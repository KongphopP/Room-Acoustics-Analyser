-- Room Acoustics Analyser: DuckDB schema
-- Rule: data we were GIVEN (metadata) is kept apart from values we MEASURED
-- ourselves (features, ISO 3382), and data-quality results get their own table.
-- Every RIR is resampled to 16 kHz at load time (decision 2026-11-19).

-- 1 row = 1 Arni panel configuration (from combinations_setup.csv)
CREATE TABLE arni_setups (
    num_comb     INTEGER PRIMARY KEY,       -- row number in the CSV = numComb in file names
    num_closed   INTEGER NOT NULL CHECK (num_closed BETWEEN 0 AND 55),
    panels_open  VARCHAR NOT NULL CHECK (length(panels_open) = 55),  -- 55 chars of '0'/'1', '1' = open
    CHECK (num_closed + length(replace(panels_open, '0', '')) = 55)  -- the rule we proved on 11,130 files
);

-- 1 row = 1 room impulse response (one channel of one file)
CREATE TABLE rirs (
    rir_uid       VARCHAR PRIMARY KEY,      -- our own id, e.g. 'rirmega_1k_rir_000000_ch0', 'arni_c442_m1_s1'
    dataset       VARCHAR NOT NULL CHECK (dataset IN ('rirmega', 'arni', 'pra_sim')),
    split         VARCHAR NOT NULL CHECK (split IN ('train', 'test')),
    source_file   VARCHAR NOT NULL,         -- path inside data/raw/, so we can always trace back
    channel       INTEGER NOT NULL DEFAULT 0,
    fs_original   INTEGER NOT NULL,         -- sample rate of the file on disk (16000 or 44100)
    fs            INTEGER NOT NULL DEFAULT 16000 CHECK (fs = 16000),  -- after resampling
    n_samples     INTEGER NOT NULL CHECK (n_samples > 0),             -- after resampling
    room_type     VARCHAR,                  -- e.g. 'shoebox'; NULL if unknown
    room_l        DOUBLE CHECK (room_l > 0),  -- metres
    room_w        DOUBLE CHECK (room_w > 0),
    room_h        DOUBLE CHECK (room_h > 0),
    src_x DOUBLE, src_y DOUBLE, src_z DOUBLE,
    mic_x DOUBLE, mic_y DOUBLE, mic_z DOUBLE,
    distance_m    DOUBLE CHECK (distance_m > 0),
    arni_num_comb INTEGER REFERENCES arni_setups (num_comb)   -- only for Arni rows, else NULL
);

-- 1 row = 1 RIR x 1 octave band. Values WE measured with one ISO 3382 code path.
-- band_hz = 0 means broadband (no octave filter).
CREATE TABLE features (
    rir_uid          VARCHAR NOT NULL REFERENCES rirs (rir_uid),
    band_hz          INTEGER NOT NULL CHECK (band_hz IN (0, 125, 250, 500, 1000, 2000, 4000)),
    edt_s            DOUBLE,   -- seconds; NULL if the decay is too shallow to measure
    t20_s            DOUBLE,
    t30_s            DOUBLE,
    c50_db           DOUBLE,
    c80_db           DOUBLE,
    drr_db           DOUBLE,
    peak_to_noise_db DOUBLE,
    PRIMARY KEY (rir_uid, band_hz)
);

-- 1 row = 1 RIR x 1 band x 1 source. Labels that came WITH a dataset
-- (e.g. RIR-Mega Sabine RT60). For comparison in EDA only, never a training target.
CREATE TABLE reference_labels (
    rir_uid  VARCHAR NOT NULL REFERENCES rirs (rir_uid),
    band_hz  INTEGER NOT NULL,
    source   VARCHAR NOT NULL,          -- e.g. 'rirmega_sabine'
    metric   VARCHAR NOT NULL,          -- e.g. 'rt60_s'
    value    DOUBLE,
    PRIMARY KEY (rir_uid, band_hz, source, metric)
);

-- 1 row = 1 RIR x 1 check. Used to filter data before training and to report counts.
CREATE TABLE data_checks (
    rir_uid     VARCHAR NOT NULL REFERENCES rirs (rir_uid),
    check_name  VARCHAR NOT NULL,       -- e.g. 't30_dynamic_range_db', 'md5_ok'
    passed      BOOLEAN NOT NULL,
    value       DOUBLE,                 -- the measured number (NULL for yes/no checks)
    threshold   DOUBLE,                 -- the limit we compared against
    checked_at  TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (rir_uid, check_name)
);

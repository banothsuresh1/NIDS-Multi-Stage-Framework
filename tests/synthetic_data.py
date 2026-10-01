"""Generate a synthetic CIC-IDS2017-shaped dataset.

Used by the test-suite, and by anyone who wants to exercise the pipeline before
obtaining the real CIC-IDS2017 release. The generated CSVs mirror the public
layout: the 78 CICFlowMeter feature columns (including the duplicated
``Fwd Header Length.1``), the identifier columns of the ``TrafficLabelling``
release, the same raw label spellings, and the same 12-hour timestamp format
with afternoon files unmarked.

Class-conditional feature distributions are deliberately separable so that
feature selection and the classifiers have real signal to find -- the point is
to exercise the code paths and the leakage guards, not to stand in for the
real data.

CLI::

    python tests/synthetic_data.py --out_dir data/synthetic --rows 60000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# The 78 CICFlowMeter feature columns of the public release, in order.
FEATURE_COLUMNS: List[str] = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

IDENTIFIER_COLUMNS = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Protocol", "Timestamp",
]

# (raw label, relative weight, port pool, duration scale, pkt scale, byte-rate scale)
CLASS_PROFILES: Dict[str, Tuple[float, List[int], float, float, float]] = {
    "BENIGN":                     (0.60, [80, 443, 53, 22, 8080], 1.0, 1.0, 1.0),
    "DoS Hulk":                   (0.10, [80],                    0.05, 8.0, 40.0),
    "PortScan":                   (0.08, [21, 22, 23, 25, 80, 110, 139, 445], 0.01, 0.2, 0.3),
    "DDoS":                       (0.07, [80, 443],               0.02, 9.0, 60.0),
    "DoS GoldenEye":              (0.03, [80],                    0.20, 4.0, 12.0),
    "FTP-Patator":                (0.025, [21],                   3.0, 1.5, 0.4),
    "SSH-Patator":                (0.02, [22],                    5.0, 1.6, 0.3),
    "DoS slowloris":              (0.02, [80],                    20.0, 0.6, 0.05),
    "DoS Slowhttptest":           (0.02, [80],                    18.0, 0.7, 0.06),
    "Bot":                        (0.012, [8080, 443],            2.0, 1.2, 0.8),
    "Web Attack \x96 Brute Force": (0.009, [80, 8080],            2.5, 1.3, 1.1),
    "Web Attack \x96 XSS":        (0.005, [80, 8080],             2.2, 1.1, 1.0),
    "Infiltration":               (0.0012, [444, 8080],           6.0, 1.4, 1.3),
    "Web Attack \x96 Sql Injection": (0.0008, [80],               2.8, 1.2, 1.0),
    "Heartbleed":                 (0.0005, [444],                 50.0, 2.0, 3.0),
}

DAY_FILES = [
    ("Monday-WorkingHours.pcap_ISCX.csv", False),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", False),
    ("Wednesday-workingHours.pcap_ISCX.csv", False),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", False),
    ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", True),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv", False),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", True),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", True),
]


def _sample_class_rows(label: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw ``n`` class-conditional feature rows for ``label``."""
    _, ports, dur_s, pkt_s, rate_s = CLASS_PROFILES[label]

    duration = np.abs(rng.lognormal(mean=np.log(1e6 * dur_s + 1), sigma=1.0, size=n))
    fwd_pkts = np.maximum(1, rng.poisson(lam=8 * pkt_s, size=n))
    bwd_pkts = np.maximum(0, rng.poisson(lam=6 * pkt_s * 0.8, size=n))
    fwd_len = np.abs(rng.normal(300 * pkt_s, 90, size=n)) * fwd_pkts
    bwd_len = np.abs(rng.normal(250 * pkt_s, 80, size=n)) * bwd_pkts
    byte_rate = (fwd_len + bwd_len) / np.maximum(duration / 1e6, 1e-6) * rate_s
    pkt_rate = (fwd_pkts + bwd_pkts) / np.maximum(duration / 1e6, 1e-6)

    df = pd.DataFrame(index=range(n))
    df["Destination Port"] = rng.choice(ports, size=n)
    df["Flow Duration"] = duration
    df["Total Fwd Packets"] = fwd_pkts
    df["Total Backward Packets"] = bwd_pkts
    df["Total Length of Fwd Packets"] = fwd_len
    df["Total Length of Bwd Packets"] = bwd_len
    df["Fwd Packet Length Max"] = fwd_len / fwd_pkts * rng.uniform(1.0, 1.6, n)
    df["Fwd Packet Length Min"] = fwd_len / fwd_pkts * rng.uniform(0.2, 0.9, n)
    df["Fwd Packet Length Mean"] = fwd_len / fwd_pkts
    df["Fwd Packet Length Std"] = np.abs(rng.normal(40 * pkt_s, 15, n))
    df["Bwd Packet Length Max"] = np.where(bwd_pkts > 0, bwd_len / np.maximum(bwd_pkts, 1) * 1.4, 0)
    df["Bwd Packet Length Min"] = np.where(bwd_pkts > 0, bwd_len / np.maximum(bwd_pkts, 1) * 0.3, 0)
    df["Bwd Packet Length Mean"] = bwd_len / np.maximum(bwd_pkts, 1)
    df["Bwd Packet Length Std"] = np.abs(rng.normal(35 * pkt_s, 12, n))
    df["Flow Bytes/s"] = byte_rate
    df["Flow Packets/s"] = pkt_rate

    iat_mean = duration / np.maximum(fwd_pkts + bwd_pkts, 1)
    df["Flow IAT Mean"] = iat_mean
    df["Flow IAT Std"] = np.abs(rng.normal(0, 1, n)) * iat_mean
    df["Flow IAT Max"] = iat_mean * rng.uniform(1.5, 6.0, n)
    df["Flow IAT Min"] = iat_mean * rng.uniform(0.0, 0.5, n)
    df["Fwd IAT Total"] = duration * rng.uniform(0.4, 1.0, n)
    df["Fwd IAT Mean"] = df["Fwd IAT Total"] / np.maximum(fwd_pkts, 1)
    df["Fwd IAT Std"] = np.abs(rng.normal(0, 1, n)) * df["Fwd IAT Mean"]
    df["Fwd IAT Max"] = df["Fwd IAT Mean"] * rng.uniform(1.5, 5.0, n)
    df["Fwd IAT Min"] = df["Fwd IAT Mean"] * rng.uniform(0.0, 0.4, n)
    df["Bwd IAT Total"] = duration * rng.uniform(0.2, 0.9, n)
    df["Bwd IAT Mean"] = df["Bwd IAT Total"] / np.maximum(bwd_pkts, 1)
    df["Bwd IAT Std"] = np.abs(rng.normal(0, 1, n)) * df["Bwd IAT Mean"]
    df["Bwd IAT Max"] = df["Bwd IAT Mean"] * rng.uniform(1.5, 5.0, n)
    df["Bwd IAT Min"] = df["Bwd IAT Mean"] * rng.uniform(0.0, 0.4, n)

    is_scan = label == "PortScan"
    df["Fwd PSH Flags"] = rng.binomial(1, 0.3 if not is_scan else 0.02, n)
    df["Bwd PSH Flags"] = 0
    df["Fwd URG Flags"] = rng.binomial(1, 0.02, n)
    df["Bwd URG Flags"] = 0
    df["Fwd Header Length"] = fwd_pkts * 32
    df["Bwd Header Length"] = bwd_pkts * 32
    df["Fwd Packets/s"] = fwd_pkts / np.maximum(duration / 1e6, 1e-6)
    df["Bwd Packets/s"] = bwd_pkts / np.maximum(duration / 1e6, 1e-6)

    all_mean = (fwd_len + bwd_len) / np.maximum(fwd_pkts + bwd_pkts, 1)
    df["Min Packet Length"] = all_mean * rng.uniform(0.1, 0.6, n)
    df["Max Packet Length"] = all_mean * rng.uniform(1.2, 2.5, n)
    df["Packet Length Mean"] = all_mean
    df["Packet Length Std"] = np.abs(rng.normal(50 * pkt_s, 20, n))
    df["Packet Length Variance"] = df["Packet Length Std"] ** 2
    df["FIN Flag Count"] = rng.binomial(1, 0.25, n)
    df["SYN Flag Count"] = rng.binomial(1, 0.9 if is_scan else 0.3, n)
    df["RST Flag Count"] = rng.binomial(1, 0.7 if is_scan else 0.05, n)
    df["PSH Flag Count"] = rng.binomial(1, 0.4, n)
    df["ACK Flag Count"] = rng.binomial(1, 0.8, n)
    df["URG Flag Count"] = rng.binomial(1, 0.03, n)
    df["CWE Flag Count"] = 0            # constant in the real release too
    df["ECE Flag Count"] = rng.binomial(1, 0.01, n)
    df["Down/Up Ratio"] = bwd_pkts / np.maximum(fwd_pkts, 1)
    df["Average Packet Size"] = all_mean * rng.uniform(0.9, 1.1, n)
    df["Avg Fwd Segment Size"] = df["Fwd Packet Length Mean"]
    df["Avg Bwd Segment Size"] = df["Bwd Packet Length Mean"]
    df["Fwd Header Length.1"] = df["Fwd Header Length"]      # duplicate column
    for col in ["Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
                "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"]:
        df[col] = 0                                           # constant, as in the release
    df["Subflow Fwd Packets"] = fwd_pkts
    df["Subflow Fwd Bytes"] = fwd_len
    df["Subflow Bwd Packets"] = bwd_pkts
    df["Subflow Bwd Bytes"] = bwd_len
    # -1 is the documented "not observed" sentinel for the window-size features.
    df["Init_Win_bytes_forward"] = np.where(
        rng.random(n) < 0.1, -1, rng.integers(0, 65535, n))
    df["Init_Win_bytes_backward"] = np.where(
        rng.random(n) < 0.2, -1, rng.integers(0, 65535, n))
    df["act_data_pkt_fwd"] = np.maximum(0, fwd_pkts - rng.integers(0, 3, n))
    df["min_seg_size_forward"] = rng.choice([20, 32, 40], size=n)
    active = duration * rng.uniform(0.1, 0.6, n)
    df["Active Mean"], df["Active Std"] = active, np.abs(rng.normal(0, 1, n)) * active
    df["Active Max"], df["Active Min"] = active * 1.5, active * 0.4
    idle = duration * rng.uniform(0.0, 0.5, n)
    df["Idle Mean"], df["Idle Std"] = idle, np.abs(rng.normal(0, 1, n)) * idle
    df["Idle Max"], df["Idle Min"] = idle * 1.6, idle * 0.3

    # Reproduce the real release's data-quality warts.
    rate_cols = ["Flow Bytes/s", "Flow Packets/s"]
    for col in rate_cols:
        mask = rng.random(n) < 0.004
        df.loc[mask, col] = np.inf
    mask = rng.random(n) < 0.003
    df.loc[mask, "Flow Bytes/s"] = np.nan

    return df[FEATURE_COLUMNS]


def generate(out_dir: str | Path, total_rows: int = 60000, seed: int = 7,
             include_identifiers: bool = True, n_hosts: int = 20,
             n_servers: int = 5, file_window_s: int = 1200) -> Path:
    """Write the synthetic daily CSVs and return the output directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    labels = list(CLASS_PROFILES)
    weights = np.array([CLASS_PROFILES[l][0] for l in labels], dtype=float)
    weights = weights / weights.sum()
    counts = np.maximum(12, (weights * total_rows).astype(int))

    frames = []
    for label, n in zip(labels, counts):
        block = _sample_class_rows(label, int(n), rng)
        block["Label"] = label
        frames.append(block)
    data = pd.concat(frames, ignore_index=True)
    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(data)
    if include_identifiers:
        # A small host population so that the symmetric communication key
        # recurs often enough to form multi-flow sessions, as in the real capture.
        # Attacks are launched from a FEW dedicated hosts rather than spread
        # uniformly, as in the real capture (one attacker runs the port scan,
        # one box is the bot, and so on). Without this host affinity the
        # communication graph carries no class signal at all and the GNN has
        # nothing to learn from structure.
        src = np.array([f"192.168.10.{i}" for i in rng.integers(1, n_hosts + 1, n)])
        dst = np.array([f"172.16.0.{i}" for i in rng.integers(1, n_servers + 1, n)])
        labels_arr = data["Label"].to_numpy()
        attacker_of = {}
        for k, lab in enumerate(sorted(set(labels_arr) - {"BENIGN"})):
            attacker_of[lab] = f"192.168.10.{(n_hosts + 1 + k)}"
        for lab, host in attacker_of.items():
            mask = labels_arr == lab
            # 85% of a family's flows come from its own attacker host.
            own = mask & (rng.random(n) < 0.85)
            src[own] = host
        data.insert(0, "Flow ID", [f"F{i:08d}" for i in range(n)])
        data.insert(1, "Source IP", src)
        data.insert(2, "Source Port", rng.integers(1024, 65535, n))
        data.insert(3, "Destination IP", dst)
        data.insert(4, "Protocol", rng.choice([6, 17, 0], size=n, p=[0.85, 0.13, 0.02]))

    # Spread rows across the daily files, newest-last, with 12-hour timestamps.
    per_file = int(np.ceil(n / len(DAY_FILES)))
    start = pd.Timestamp("2017-07-03 09:00:00")
    written = []
    for d, (fname, is_afternoon) in enumerate(DAY_FILES):
        chunk = data.iloc[d * per_file:(d + 1) * per_file].copy()
        if chunk.empty:
            continue
        # Place each file on its real weekday, morning or afternoon, so the
        # eight capture windows are well separated in time (as in the release).
        weekday = ["monday", "tuesday", "wednesday", "thursday", "friday"]
        day_idx = next((i for i, w in enumerate(weekday) if w in fname.lower()), d)
        base = start + pd.Timedelta(days=day_idx) + (
            pd.Timedelta(hours=5) if is_afternoon else pd.Timedelta(0))
        if fname.endswith("DDos.pcap_ISCX.csv"):
            base += pd.Timedelta(hours=2)        # second Friday-afternoon window
        # Dense in time (~4 flows/s), matching the real capture's rate, so that
        # the session idle timeout produces multi-flow sessions.
        ts = base + pd.to_timedelta(np.sort(rng.integers(0, file_window_s, len(chunk))), unit="s")
        if include_identifiers:
            # 12-hour clock WITHOUT am/pm, exactly as the public CSVs store it.
            chunk["Timestamp"] = pd.Series(ts).dt.strftime("%d/%m/%Y %I:%M:%S").values
            cols = IDENTIFIER_COLUMNS + [c for c in FEATURE_COLUMNS if c != "Protocol"] + ["Label"]
            chunk = chunk[[c for c in cols if c in chunk.columns]]
        path = out / fname
        # The real files carry a leading space on most header names.
        chunk.columns = [
            c if c in ("Flow ID", "Label", "Destination Port") else " " + c
            for c in chunk.columns
        ]
        chunk.to_csv(path, index=False, encoding="latin-1")
        written.append(path)

    print(f"wrote {len(written)} CSV file(s), {n} rows total -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic CIC-IDS2017-shaped dataset")
    ap.add_argument("--out_dir", default="data/synthetic")
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no_identifiers", action="store_true",
                    help="emit the MachineLearningCVE layout (no Flow ID / IPs / Timestamp)")
    args = ap.parse_args()
    generate(args.out_dir, args.rows, args.seed, include_identifiers=not args.no_identifiers)


if __name__ == "__main__":
    main()

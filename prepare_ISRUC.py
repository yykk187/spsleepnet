import argparse
import glob
import math
import ntpath
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from mne.io import read_raw_edf
# from scipy.signal import resample  # optional

# Label values
W, N1, N2, N3, REM, UNKNOWN = 0, 1, 2, 3, 4, 5
stage_dict = {"W": W, "N1": N1, "N2": N2, "N3": N3, "REM": REM, "UNKNOWN": UNKNOWN}

ann2label = {
    "W": 0, "SLEEP-S0": 0,
    "N1": 1, "SLEEP-S1": 1,
    "N2": 2, "SLEEP-S2": 2,
    "N3": 3, "SLEEP-S3": 3,
    "N4": 3,
    "R": 4, "SLEEP-REM": 4,
    "U": 5, "Movement time": 5
}

position_dict = {
    "N": 0,  # Neutral
    "L": 1,  # Left
    "R": 2,  # Right
    "P": 3,  # Prone
    "B": 4   # Back
}

EPOCH_SEC_SIZE = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/media/qiu/DataDisk1/yue/data/ISRUC",
                        help="Parent folder with subfolders of edf + xlsx files.")
    parser.add_argument("--output_dir", type=str, default="/media/qiu/DataDisk1/yue/data/ISRUCNPZ",
                        help="Output directory to save .npz files.")
    parser.add_argument("--select_ch", type=str, default="F4-M1",
                        help="Channel name to extract, e.g., F4-M1")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    folders = [d for d in glob.glob(os.path.join(args.data_dir, '*')) if os.path.isdir(d)]

    for folder in folders:
        edf_files = glob.glob(os.path.join(folder, "*.edf"))
        xlsx_files = glob.glob(os.path.join(folder, "*1.xlsx"))

        if not edf_files or not xlsx_files:
            print(f"跳过文件夹（无edf或xlsx）：{folder}")
            continue

        edf_file = edf_files[0]
        xlsx_file = xlsx_files[0]

        raw = read_raw_edf(edf_file, preload=True, stim_channel=None, verbose='ERROR')
        print(f"\n处理文件: {edf_file}")

        if args.select_ch not in raw.ch_names:
            print(f"通道 '{args.select_ch}' 不在 {edf_file} 中，跳过。")
            continue

        sampling_rate = raw.info['sfreq']
        raw_ch_df = raw.to_data_frame()[args.select_ch].to_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))

        h_raw = {
            "meas_date": raw.info.get("meas_date"),
            "sfreq": raw.info.get("sfreq"),
            "nchan": raw.info.get("nchan"),
            "ch_names": raw.ch_names,
            "subject_info": raw.info.get("subject_info"),
            "highpass": raw.info.get("highpass"),
            "lowpass": raw.info.get("lowpass")
        }

        df = pd.read_excel(xlsx_file)
        print(f"读取注释: {xlsx_file}")

        ann = []
        positions = []
        onset_time = 0.0

        for _, row in df.iterrows():
            sleep_stage = str(row.iloc[1]).strip()
            position = str(row.iloc[5]).strip()

            # 跳过未知体位
            if position not in position_dict:
                print(f"忽略未知体位: {position} @ {onset_time}s")
                continue

            ann.append([onset_time, EPOCH_SEC_SIZE, sleep_stage])
            positions.append(position_dict[position])
            onset_time += EPOCH_SEC_SIZE

        remove_idx, labels, label_idx = [], [], []
        for a in ann:
            onset_sec, duration_sec, ann_char = a
            label = ann2label.get(ann_char, UNKNOWN)
            idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
            if label != UNKNOWN:
                labels.append(np.ones(int(duration_sec / EPOCH_SEC_SIZE), dtype=int) * label)
                label_idx.append(idx)
            else:
                remove_idx.append(idx)
                print(f"忽略未知标签: {ann_char} @ {onset_sec}s")

        if not labels or not label_idx:
            print(f"跳过文件（无有效标签）：{edf_file}")
            continue

        labels = np.hstack(labels)
        remove_idx = np.hstack(remove_idx) if remove_idx else np.array([], dtype=int)
        label_idx = np.hstack(label_idx)

        select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
        select_idx = np.intersect1d(select_idx, label_idx)

        if len(label_idx) > len(select_idx):
            extra_idx = np.setdiff1d(label_idx, select_idx)
            if np.all(extra_idx > select_idx[-1]):
                labels = labels[:len(select_idx) // int(sampling_rate * EPOCH_SEC_SIZE)]

        raw_ch = raw_ch_df.values[select_idx]
        n_epochs = int(len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate))
        n_samples = int(n_epochs * EPOCH_SEC_SIZE * sampling_rate)
        raw_ch = raw_ch[:n_samples]
        positions = positions[:n_epochs]

        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels[:n_epochs].astype(np.int32)
        p = np.array(positions).astype(np.int32)

        assert len(x) == len(y) == len(p)

        nw_idx = np.where(y != stage_dict["W"])[0]
        if len(nw_idx) == 0:
            print(f"未找到非清醒睡眠阶段，跳过：{edf_file}")
            continue

        start_idx = max(nw_idx[0] - 30 * 2, 0)
        end_idx = min(nw_idx[-1] + 30 * 2, len(y) - 1)
        keep_idx = np.arange(start_idx, end_idx + 1)

        x, y, p = x[keep_idx], y[keep_idx], p[keep_idx]

        filename = ntpath.basename(edf_file).replace(".edf", ".npz")
        np.savez(os.path.join(args.output_dir, filename),
                 x=x, y=y, p=p, fs=sampling_rate,
                 ch_label=args.select_ch, header_raw=h_raw)

        print(f"保存: {filename} | 数据维度: {x.shape}, 标签: {y.shape}, 体位: {p.shape}")


if __name__ == "__main__":
    main()

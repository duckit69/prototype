import pickle
from pathlib import Path
import pandas as pd
import numpy as np

class WESADPhase2WindowBuilder:
    """Build window-level data from subject-level WESAD parquet.

    Phase 2:
    - load the subject-level parquet created in Phase 1
    - extract wrist EDA and original label streams
    - build fixed windows on the wrist timeline
    - align labels to each window using majority vote
    - map labels to binary classes
    - save a window-level parquet for feature extraction
    """

    def __init__(self, input_parquet: str, output_dir: str = "output"):
        self.input_parquet = Path(input_parquet)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.windows = []

    def load_subject_table(self):
        self.df = pd.read_parquet(self.input_parquet)
        return self.df

    @staticmethod
    def map_label_to_binary(label: int):
        # WESAD: 2 = stress, 1/3/4 = non-stress, others ignored
        if label == 2:
            return 1
        if label in (1, 3, 4):
            return 0
        return -1

    @staticmethod
    def dominant_label(labels):
        labels = np.asarray(labels, dtype=np.int32)
        labels = labels[labels != -1]
        if len(labels) == 0:
            return -1
        vals, counts = np.unique(labels, return_counts=True)
        return int(vals[np.argmax(counts)])

    @staticmethod
    def iter_windows(arr, window_size, step_size):
        n = len(arr)
        start = 0
        while start + window_size <= n:
            end = start + window_size
            yield start, end, arr[start:end]
            start += step_size

    def build_windows(self, window_seconds=60, step_seconds=30, eda_fs=4, label_fs=700):
        self.load_subject_table()
        self.windows = []
        window_size = int(window_seconds * eda_fs)
        step_size = int(step_seconds * eda_fs)

        for _, row in self.df.iterrows():
            subject = row["subject"]
            eda = np.array(row["eda_raw"], dtype=np.float32)
            labels = np.array(row["label_raw"], dtype=np.int32)

            # We create windows on the EDA timeline.
            # Labels are reduced to a matching window label by majority vote using appropriate sampling rates.
            for w_id, (s, e, eda_win) in enumerate(self.iter_windows(eda, window_size, step_size)):
                # Calculate the start and end time of the window in seconds
                start_sec = s / eda_fs
                end_sec = e / eda_fs
                
                # Convert the time in seconds to label array indices
                s_label = int(start_sec * label_fs)
                e_label = int(end_sec * label_fs)
                
                # Ensure we don't index beyond the available labels
                if s_label >= len(labels):
                    break
                e_label = min(e_label, len(labels))
                
                label_win = labels[s_label:e_label]
                dom = self.dominant_label(label_win)
                bin_lab = self.map_label_to_binary(dom)
                if bin_lab == -1:
                    continue
                self.windows.append({
                    "subject": subject,
                    "window_id": w_id,
                    "start_idx": s,
                    "end_idx": e,
                    "eda_window": eda_win.tolist(),
                    "raw_label_window": label_win.tolist(),
                    "dominant_label": dom,
                    "binary_label": bin_lab,
                    "window_seconds": window_seconds,
                    "step_seconds": step_seconds,
                    "eda_fs": eda_fs,
                    "n_samples": len(eda_win),
                })

        return pd.DataFrame(self.windows)

    def save_windows(self, filename="wesad_windows_phase2.parquet", **kwargs):
        dfw = self.build_windows(**kwargs)
        out = self.output_dir / filename
        dfw.to_parquet(out, index=False)
        return dfw, out

    def summary(self, dfw):
        return dfw.groupby(["subject", "binary_label"]).size().unstack(fill_value=0)

if __name__ == '__main__':
    builder = WESADPhase2WindowBuilder('output/wesad_subject_level_raw.parquet')
    dfw, out = builder.save_windows(window_seconds=60, step_seconds=30, eda_fs=4)
    print(builder.summary(dfw))
    print(f'Saved to: {out}')
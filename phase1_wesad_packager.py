import pickle
from pathlib import Path
import pandas as pd
import numpy as np

class WESADRawPackager:
    """Load WESAD subject pickle files and pack them into one parquet file.

    This first phase keeps the data at subject level:
    - one row per subject
    - raw wrist EDA stored as a nested list
    - raw labels stored as a nested list
    - basic metadata for quick inspection
    """

    def __init__(self, root_dir: str, output_dir: str = "output"):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.subject_dirs = []
        self.rows = []

    def discover_subjects(self):
        self.subject_dirs = sorted([
            p for p in self.root_dir.iterdir()
            if p.is_dir() and p.name.startswith("S")
        ])
        return self.subject_dirs

    def load_subject_pickle(self, subject_dir: Path):
        pkl_path = subject_dir / f"{subject_dir.name}.pkl"
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f, encoding="latin1")
        eda = np.asarray(obj["signal"]["wrist"]["EDA"]).reshape(-1)
        labels = np.asarray(obj["label"]).reshape(-1)
        return obj, eda, labels

    @staticmethod
    def safe_to_list(arr):
        return np.asarray(arr).reshape(-1).tolist()

    def build_rows(self):
        self.discover_subjects()
        self.rows = []
        for subject_dir in self.subject_dirs:
            obj, eda, labels = self.load_subject_pickle(subject_dir)
            self.rows.append({
                "subject": subject_dir.name,
                "eda_raw": self.safe_to_list(eda),
                "label_raw": self.safe_to_list(labels),
                "eda_len": int(len(eda)),
                "label_len": int(len(labels)),
                "signal_keys": list(obj["signal"].keys()),
                "wrist_keys": list(obj["signal"]["wrist"].keys()),
            })
        return pd.DataFrame(self.rows)

    def save_parquet(self, filename: str = "wesad_subject_level_raw.parquet"):
        df = self.build_rows()
        out = self.output_dir / filename
        df.to_parquet(out, index=False)
        return df, out

    def summary(self, df: pd.DataFrame):
        return df[["subject", "eda_len", "label_len"]].copy()

if __name__ == '__main__':
    packager = WESADRawPackager(root_dir='data')
    df, out = packager.save_parquet()
    print(packager.summary(df).to_string(index=False))
    print(f'Saved parquet to: {out}')
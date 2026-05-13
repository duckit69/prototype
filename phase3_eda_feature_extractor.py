import neurokit2 as nk
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import welch
import warnings


class WESADEDAFeatureExtractor:
    """Extract EDA features from window-level WESAD parquet (Phase 2 output).

    Phase 3:
    - load the window-level parquet created in Phase 2
    - for each EDA window, compute tonic/phasic features via NeuroKit2
    - compute statistical time-domain features on raw EDA
    - compute frequency-domain features (spectral power bands)
    - save a feature-level parquet for modelling
    """

    SAMPLING_RATE = 4  # Hz (wrist EDA)

    def __init__(self, input_parquet: str, output_dir: str = "output"):
        self.input_parquet = Path(input_parquet)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.all_features: list[dict] = []

    # ------------------------------------------------------------------

    def load_windows(self) -> pd.DataFrame:
        self.df = pd.read_parquet(self.input_parquet)
        return self.df

    # ------------------------------------------------------------------

    def _tonic_phasic_features(self, eda_signal) -> dict:
        """Return tonic/phasic features for a single EDA window."""
        try:
            df_result = nk.eda_phasic(
                eda_signal, method="smoothmedian", sampling_rate=self.SAMPLING_RATE
            )
            phasic = df_result["EDA_Phasic"].values
            tonic = df_result["EDA_Tonic"].values

            # Tonic
            mu_tonic = np.mean(tonic)
            slope_tonic = np.polyfit(np.arange(len(tonic)), tonic, 1)[0]
            sigma_tonic = np.std(tonic)

            # Phasic peaks
            peak_signals, info = nk.eda_peaks(
                phasic,
                sampling_rate=self.SAMPLING_RATE,
                method="neurokit",
                amplitude_min=0.01,
            )
            nscr = len(info["SCR_Peaks"]) if "SCR_Peaks" in info else 0
            mu_ampl = np.nanmean(info["SCR_Amplitude"]) if nscr > 0 else 0.0
            max_ampl = np.nanmax(info["SCR_Amplitude"]) if nscr > 0 else 0.0
            rise_time = (
                np.nanmean(info["SCR_RiseTime"])
                if "SCR_RiseTime" in info and len(info["SCR_RiseTime"]) > 0
                else 0.0
            )
            decay_time = (
                np.nanmean(info["SCR_RecoveryTime"])
                if "SCR_RecoveryTime" in info and len(info["SCR_RecoveryTime"]) > 0
                else 0.0
            )

        except Exception:
            mu_tonic = slope_tonic = sigma_tonic = np.nan
            nscr = 0
            mu_ampl = max_ampl = rise_time = decay_time = np.nan

        return dict(
            mu_tonic=mu_tonic,
            slope_tonic=slope_tonic,
            sigma_tonic=sigma_tonic,
            nscr=nscr,
            mu_ampl=mu_ampl,
            max_ampl=max_ampl,
            rise_time=rise_time,
            decay_time=decay_time,
        )

    @staticmethod
    def _statistical_features(eda_signal) -> dict:
        """Return statistical time-domain features for a single EDA window."""
        return dict(
            eda_std=np.std(eda_signal),
            eda_mean=np.mean(eda_signal),
            eda_median=np.median(eda_signal),
            eda_p25=np.percentile(eda_signal, 25),
            eda_p75=np.percentile(eda_signal, 75),
        )

    def _frequency_features(self, eda_signal) -> dict:
        """Return frequency-domain (spectral power) features for a single EDA window."""
        nperseg = min(256, len(eda_signal))
        freqs, psd = welch(eda_signal, fs=self.SAMPLING_RATE, nperseg=nperseg)

        idx_0_02 = (freqs >= 0) & (freqs <= 0.2)
        power_0_02 = (
            np.trapezoid(psd[idx_0_02], freqs[idx_0_02]) if np.any(idx_0_02) else 0.0
        )

        idx_02_05 = (freqs >= 0.2) & (freqs <= 0.5)
        power_02_05 = (
            np.trapezoid(psd[idx_02_05], freqs[idx_02_05])
            if np.any(idx_02_05)
            else 0.0
        )

        return dict(power_0_02=power_0_02, power_02_05=power_02_05)

    # Extraction
    # ------------------------------------------------------------------

    def extract_features(self) -> pd.DataFrame:
        """Iterate over all windows and extract features for every subject."""
        self.load_windows()
        self.all_features = []

        for subject, grp in self.df.groupby("subject"):
            print(f"\nProcessing Subject: {subject} ({len(grp)} windows)...")
            subject_features = []

            for idx, row in grp.iterrows():
                eda_signal = row["eda_window"]

                feat = {
                    "subject": subject,
                    "window_id": row.get("window_id", idx),
                    "binary_label": row.get("binary_label", np.nan),
                }
                feat.update(self._tonic_phasic_features(eda_signal))
                feat.update(self._statistical_features(eda_signal))
                feat.update(self._frequency_features(eda_signal))

                subject_features.append(feat)
                self.all_features.append(feat)

            # Per-subject summary
            subj_df = pd.DataFrame(subject_features)
            print(f"  -> Extracted {len(subj_df)} windows")
            print(f"  -> Mean SCL (mu_tonic): {subj_df['mu_tonic'].mean():.4f}")
            print(f"  -> Avg SCR Peaks (nscr): {subj_df['nscr'].mean():.2f}")
            print(
                f"  -> Avg Spectral Power (0.2-0.5 Hz): {subj_df['power_02_05'].mean():.4f}"
            )

        return pd.DataFrame(self.all_features)

    def save_features(self, filename: str = "wesad_eda_features_phase3.parquet") -> tuple[pd.DataFrame, Path]:
        features_df = self.extract_features()
        out = self.output_dir / filename
        features_df.to_parquet(out, index=False)
        return features_df, out

    def summary(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Return a high-level summary of extracted features."""
        return (
            features_df.groupby("subject")[
                ["mu_tonic", "nscr", "power_02_05"]
            ]
            .mean()
            .round(4)
        )


if __name__ == "__main__":
    extractor = WESADEDAFeatureExtractor("output/wesad_windows_phase2.parquet")
    features_df, out = extractor.save_features()
    print(f"\n==========================================")
    print(f"Successfully processed {len(features_df)} total windows.")
    print(f"Features DataFrame shape: {features_df.shape}")
    print(f"Saved to: {out}")
    print("\nPer-subject summary:")
    print(extractor.summary(features_df).to_string())

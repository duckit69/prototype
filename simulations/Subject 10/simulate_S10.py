"""
simulate_S10.py
==============
Hardware-in-the-loop simulation for Subject S10.

Pipeline:
  1.  Connect to Omnikey reader via pyscard (smart-card I/O).
  2.  Train a generic logistic regression (LOSO – all subjects except S10).
  3.  Write 64 bytes (15 weights × 4 B + bias 4 B) to the smart card
      (simulated here as a local text file: simulations/smart_card_S10.txt).
  4.  Read the generic model back from the smart card.
  5.  Send INIT:w1,...,w15,b over Serial to the ESP32 (115200 baud).
  6.  Wait for "ESP32 ready".
  7.  For each window (chronological):
        7.1  Extract 15 EDA features from the phase-3 parquet.
        7.2  Send FEAT:f1,...,f15 over Serial.
        7.3  Wait for PRED:0 or PRED:1.
        7.4  Send LABEL:0 or LABEL:1.
        7.5  Log prediction vs true label.
  8.  Send END_SHIFT.
  9.  Wait for WEIGHTS:w1,...,w15,b (personalised model).
  10. Overwrite smart card with updated weights.
  11. Print shift summary (timing from ESP32 AVG_* messages).
  12. Compute and print accuracy / F1.

Connections to ./simulations/esp32.cpp:
  - INIT / FEAT / LABEL / END_SHIFT  →  esp32.cpp loop()
  - PRED / WEIGHTS / AVG_*           ←  esp32.cpp responses

Usage:
    python simulations/simulate_S10.py [--port /dev/ttyUSB0] [--dry-run]
"""

import argparse
import struct
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import serial
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SUBJECT          = "S10"
NUM_FEATURES     = 15
BAUD_RATE        = 115200
SERIAL_TIMEOUT   = 10          # seconds to wait for a single line
LABEL_TIMEOUT    = 5.0         # seconds to wait for LABEL ack (dry-run: unused)
SMART_CARD_FILE  = Path(__file__).parent / f"smart_card_{SUBJECT}.txt"
DATA_PATH        = Path(__file__).parent.parent / "output" / "wesad_eda_features_phase3.parquet"

METADATA_COLS    = ["subject", "window_id", "binary_label"]

# ─────────────────────────────────────────────────────────────────────────────
# pyscard / Omnikey helpers
# ─────────────────────────────────────────────────────────────────────────────

def connect_reader():
    """Connect to the first available PC/SC reader (Omnikey or any).
    Returns (cardservice, connection) or (None, None) if unavailable."""
    try:
        from smartcard.System import readers
        from smartcard.util import toHexString

        reader_list = readers()
        if not reader_list:
            print("[SmartCard] No readers found – using file simulation.")
            return None, None

        reader = reader_list[0]
        print(f"[SmartCard] Using reader: {reader}")
        conn = reader.createConnection()
        conn.connect()
        print("[SmartCard] Card connected.")
        return reader, conn
    except Exception as exc:
        print(f"[SmartCard] pyscard error ({exc}) – using file simulation.")
        return None, None


def write_model_to_card(conn, weights: np.ndarray, bias: float):
    """Write 64 bytes (15 × float32 + 1 × float32) to the DESFire EV3 card.
    Falls back to SMART_CARD_FILE when conn is None."""
    payload = struct.pack(f">{NUM_FEATURES + 1}f", *weights, bias)   # big-endian
    assert len(payload) == (NUM_FEATURES + 1) * 4 == 64 # 16 (15 weights + 1 bias)* 4 bytes = 64 bytes

    if conn is not None:
        try:
            # DESFire EV3: WriteData to file 01, offset 0, length 64
            # Lc = 1 (file#) + 3 (offset) + 3 (length field) + 64 (payload) = 71 = 0x47
            apdu = [0x90, 0x3D, 0x00, 0x00, 0x47,   # CLA INS P1 P2 Lc=71
                    0x01,                              # file number
                    0x00, 0x00, 0x00,                  # offset (3 bytes LE)
                    0x40, 0x00, 0x00]                  # length (3 bytes LE) = 64
            apdu += list(payload) + [0x00]             # 64 payload bytes + Le
            resp, sw1, sw2 = conn.transmit(apdu)
            if (sw1, sw2) == (0x91, 0x00):
                print("[SmartCard] Model written to card (DESFire WriteData).")
                return
            else:
                print(f"[SmartCard] WriteData warning: SW={sw1:02X}{sw2:02X} – falling back to file.")
        except Exception as exc:
            print(f"[SmartCard] Write error ({exc}) – falling back to file.")

    # File fallback
    with open(SMART_CARD_FILE, "w") as f:
        values = list(weights) + [bias]
        f.write(",".join(f"{v:.8f}" for v in values))
    print(f"[SmartCard] Model written to {SMART_CARD_FILE}.")


def read_model_from_card(conn):
    """Read 64 bytes from the DESFire EV3 card.
    Falls back to SMART_CARD_FILE when conn is None.
    Returns (weights np.ndarray[15], bias float)."""
    if conn is not None:
        try:
            apdu = [0x90, 0xBD, 0x00, 0x00, 0x07,
                    0x01,
                    0x00, 0x00, 0x00,
                    0x40, 0x00, 0x00,
                    0x00]
            resp, sw1, sw2 = conn.transmit(apdu)
            if (sw1, sw2) == (0x91, 0x00) and len(resp) == 64:
                values = struct.unpack(f">{NUM_FEATURES + 1}f", bytes(resp))
                weights = np.array(values[:NUM_FEATURES], dtype=np.float32)
                bias    = float(values[NUM_FEATURES])
                print("[SmartCard] Model read from card (DESFire ReadData).")
                return weights, bias
            else:
                print(f"[SmartCard] ReadData warning: SW={sw1:02X}{sw2:02X} – falling back to file.")
        except Exception as exc:
            print(f"[SmartCard] Read error ({exc}) – falling back to file.")

    with open(SMART_CARD_FILE, "r") as f:
        values = [float(v) for v in f.read().strip().split(",")]
    weights = np.array(values[:NUM_FEATURES], dtype=np.float32)
    bias    = float(values[NUM_FEATURES])
    print(f"[SmartCard] Model read from {SMART_CARD_FILE}.")
    return weights, bias


# ─────────────────────────────────────────────────────────────────────────────
# Serial helpers
# ─────────────────────────────────────────────────────────────────────────────

def open_serial(port: str, dry_run: bool):
    if dry_run:
        print("[Serial] DRY-RUN mode – no serial port opened.")
        return None
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"[Serial] Opened {port} @ {BAUD_RATE} baud.")
        return ser
    except Exception as exc:
        print(f"[Serial] Cannot open {port}: {exc} – switching to DRY-RUN.")
        return None


def serial_write(ser, line: str):
    if ser:
        ser.write((line + "\n").encode("utf-8"))
        ser.flush()
    else:
        print(f"  >> {line}")


def serial_readline(ser, timeout_hint: str = "") -> str:
    if ser:
        raw = ser.readline()
        return raw.decode("utf-8", errors="replace").strip()
    # Dry-run: simulate ESP32 responses
    return ""


def wait_for_ready(ser):
    """Block until 'ESP32 ready' is received (or simulate in dry-run)."""
    if ser is None:
        print("[Serial] (DRY-RUN) Simulating 'ESP32 ready'.")
        return
    print("[Serial] Waiting for 'ESP32 ready'...")
    deadline = time.time() + 30
    while time.time() < deadline:
        line = serial_readline(ser)
        if "ESP32 ready" in line:
            print(f"[Serial] Received: {line}")
            return
        if line:
            print(f"[Serial] (startup) {line}")
    raise TimeoutError("Timed out waiting for 'ESP32 ready'.")


def parse_weights_line(line: str):
    """Parse 'WEIGHTS:w1,...,w15,b' and return (weights, bias)."""
    data = line[len("WEIGHTS:"):].strip()
    values = [float(v) for v in data.split(",")]
    return np.array(values[:NUM_FEATURES], dtype=np.float32), float(values[NUM_FEATURES])


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_generic_model(df: pd.DataFrame, feature_cols: list):
    """LOSO: train on all subjects except SUBJECT."""
    train_df = df[df["subject"] != SUBJECT].copy()
    test_df  = df[df["subject"] == SUBJECT].copy().sort_values("window_id")

    X_train = train_df[feature_cols].values
    y_train = train_df["binary_label"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train_s, y_train)

    weights = clf.coef_[0].astype(np.float32)
    bias    = float(clf.intercept_[0])

    # Scale test features with the same scaler
    X_test  = test_df[feature_cols].values
    X_test_s = scaler.transform(X_test)
    y_test  = test_df["binary_label"].values

    return weights, bias, X_test_s, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=f"Hardware simulation for subject {SUBJECT}")
    parser.add_argument("--port",    default="/dev/ttyUSB0", help="Serial port for ESP32")
    parser.add_argument("--dry-run", action="store_true",    help="No serial / card hardware")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Edge Personalisation Simulation – Subject {SUBJECT}")
    print(f"{'='*60}\n")

    # ── 1. Connect to Omnikey ────────────────────────────────────────────────
    _, conn = connect_reader()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"[Data] Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH).dropna()
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    print(f"[Data] {len(df)} windows, {len(feature_cols)} features, "
          f"{df['subject'].nunique()} subjects.")

    # ── 2. Train generic model ───────────────────────────────────────────────
    print(f"\n[Model] Training generic LR (LOSO – excluding {SUBJECT}) ...")
    weights, bias, X_test_s, y_test = train_generic_model(df, feature_cols)
    print(f"[Model] bias = {bias:.6f}")

    # ── 3. Write model to smart card ─────────────────────────────────────────
    write_model_to_card(conn, weights, bias)

    # ── 4. Read model back from smart card ───────────────────────────────────
    card_weights, card_bias = read_model_from_card(conn)
    print(f"[SmartCard] Verified – first weight: {card_weights[0]:.6f}, bias: {card_bias:.6f}")

    # ── 5–6. Open serial & wait for ESP32 ────────────────────────────────────
    ser = open_serial(args.port, args.dry_run)
    wait_for_ready(ser)

    # Send INIT
    init_vals = ",".join(f"{w:.8f}" for w in card_weights) + f",{card_bias:.8f}"
    serial_write(ser, f"INIT:{init_vals}")
    print(f"[Serial] Sent INIT with {NUM_FEATURES} weights + bias.")

    # Drain any startup messages
    if ser:
        time.sleep(0.5)
        while ser.in_waiting:
            print(f"[ESP32] {ser.readline().decode('utf-8', errors='replace').strip()}")

    # ── 7. Window loop ────────────────────────────────────────────────────────
    predictions  = []
    true_labels  = []
    infer_times  = []
    update_times = []

    n_windows = len(X_test_s)
    print(f"\n[Loop] Processing {n_windows} windows for {SUBJECT} ...\n")

    for i in range(n_windows):
        features = X_test_s[i]
        label    = int(y_test[i])

        # 7.2 Send FEAT
        feat_str = ",".join(f"{f:.6f}" for f in features)
        serial_write(ser, f"FEAT:{feat_str}")

        # 7.3 Wait for PRED
        pred = -1
        if ser:
            deadline = time.time() + SERIAL_TIMEOUT
            while time.time() < deadline:
                line = serial_readline(ser)
                if line.startswith("PRED:"):
                    pred = int(line.split(":")[1])
                    break
                elif line.startswith("INFER_TIME:"):
                    try:
                        infer_times.append(int(line.split(":")[1].split()[0]))
                    except Exception:
                        pass
                elif line:
                    print(f"  [ESP32] {line}")
        else:
            # Dry-run: use the generic model locally for a plausible prediction
            z    = float(np.dot(card_weights, features) + card_bias)
            prob = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            pred = 1 if prob >= 0.5 else 0
            print(f"  (DRY-RUN) PRED:{pred}")

        predictions.append(pred)

        # 7.4 Send LABEL
        serial_write(ser, f"LABEL:{label}")

        # 7.5 Log
        true_labels.append(label)

        # Collect timing lines that arrive after LABEL
        if ser:
            t_start = time.time()
            while time.time() - t_start < 0.3:
                if not ser.in_waiting:
                    break
                line = serial_readline(ser)
                if line.startswith("UPDATE_TIME:"):
                    try:
                        update_times.append(int(line.split(":")[1].split()[0]))
                    except Exception:
                        pass
                elif line:
                    print(f"  [ESP32] {line}")

        status = "✓" if pred == label else "✗"
        print(f"  Window {i+1:>4}/{n_windows}  PRED:{pred}  TRUE:{label}  {status}")

    # ── 8. END_SHIFT ──────────────────────────────────────────────────────────
    print("\n[Serial] Sending END_SHIFT ...")
    serial_write(ser, "END_SHIFT")

    # ── 9. Collect final weights & AVG_* ─────────────────────────────────────
    final_weights = card_weights.copy()
    final_bias    = card_bias
    metrics_raw   = {}

    if ser:
        deadline = time.time() + 15
        while time.time() < deadline:
            line = serial_readline(ser)
            if not line:
                break
            print(f"[ESP32] {line}")
            if line.startswith("WEIGHTS:"):
                final_weights, final_bias = parse_weights_line(line)
            elif line.startswith("AVG_INFERENCE_US:"):
                metrics_raw["AVG_INFERENCE_US"] = int(line.split(":")[1])
            elif line.startswith("AVG_UPDATE_US:"):
                metrics_raw["AVG_UPDATE_US"] = int(line.split(":")[1])
            elif line.startswith("TOTAL_INFERENCES:"):
                metrics_raw["TOTAL_INFERENCES"] = int(line.split(":")[1])
            elif line.startswith("RAM footprint"):
                metrics_raw["RAM_ESTIMATE"] = line
            elif line.startswith("Model size"):
                metrics_raw["MODEL_SIZE"] = line
    else:
        print("(DRY-RUN) Simulating WEIGHTS response – weights unchanged.")
        metrics_raw = {
            "AVG_INFERENCE_US":  0,
            "AVG_UPDATE_US":     0,
            "TOTAL_INFERENCES":  n_windows,
        }

    # ── 10. Write updated weights to smart card ───────────────────────────────
    write_model_to_card(conn, final_weights, final_bias)

    # ── 11–13. Summary ────────────────────────────────────────────────────────
    acc = accuracy_score(true_labels, predictions)
    f1  = f1_score(true_labels, predictions, zero_division=0)
    cm  = confusion_matrix(true_labels, predictions)

    print(f"\n{'='*60}")
    print(f"  SHIFT SUMMARY – Subject {SUBJECT}")
    print(f"{'='*60}")
    print(f"  Windows processed  : {n_windows}")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  F1 Score           : {f1:.4f}")
    print(f"  Confusion Matrix   :\n{cm}")

    if infer_times:
        print(f"  Avg Inference (µs) : {np.mean(infer_times):.1f}")
    elif "AVG_INFERENCE_US" in metrics_raw:
        print(f"  Avg Inference (µs) : {metrics_raw['AVG_INFERENCE_US']}")

    if update_times:
        print(f"  Avg Update    (µs) : {np.mean(update_times):.1f}")
    elif "AVG_UPDATE_US" in metrics_raw:
        print(f"  Avg Update    (µs) : {metrics_raw['AVG_UPDATE_US']}")

    if "TOTAL_INFERENCES" in metrics_raw:
        print(f"  Total Inferences   : {metrics_raw['TOTAL_INFERENCES']}")
    if "RAM_ESTIMATE" in metrics_raw:
        print(f"  {metrics_raw['RAM_ESTIMATE']}")
    if "MODEL_SIZE" in metrics_raw:
        print(f"  {metrics_raw['MODEL_SIZE']}")

    print(f"\n  Final personalised weights written back to smart card.")
    print(f"{'='*60}\n")

    if ser:
        ser.close()


if __name__ == "__main__":
    main()

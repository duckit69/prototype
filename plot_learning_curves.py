import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def simulate_incremental(X, y, generic_model, lr=0.1, k_updates=10):
    """No-replay incremental SGD with gradient clipping (10 updates per label)."""
    w = generic_model.coef_[0].copy()
    b = generic_model.intercept_[0].copy()
    
    # Calculate baseline generic accuracy for padding the start of the rolling window
    acc_gen = generic_model.score(X, y)
    
    y_pred = []
    rolling_acc = []
    correct_sofar = 0
    
    for i, (x, y_true) in enumerate(zip(X, y)):
        # predict
        p = 1/(1+np.exp(-(np.dot(w, x)+b)))
        pred = 1 if p>=0.5 else 0
        y_pred.append(pred)
        correct_sofar += (pred == y_true)
        # rolling accuracy over last 100 predictions (with baseline padding for smooth start)
        if i < 100:
            padding_correct = acc_gen * (100 - (i + 1))
            rolling_acc.append((padding_correct + correct_sofar) / 100)
        else:
            rolling_acc.append((np.array(y_pred)[-100:]==np.array(y)[:i+1][-100:]).mean())
        # update on true label (10 SGD steps, gradient clipping, no replay)
        for _ in range(k_updates):
            p = 1/(1+np.exp(-(np.dot(w, x)+b)))
            error = y_true - p
            grad_w = lr * error * x
            grad_b = lr * error
            if np.linalg.norm(grad_w) > 1.0:
                grad_w = grad_w / np.linalg.norm(grad_w)
            w += grad_w
            b += grad_b
    return np.array(y_pred), np.array(rolling_acc)

def main():
    print("Loading data...")
    df = pd.read_parquet('output/wesad_eda_features_phase3.parquet')
    df = df.dropna()
    
    metadata_cols = ['subject', 'window_id', 'binary_label']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    target_subjects = ['S10', 'S14', 'S17']
    results = {}
    
    for test_subject in target_subjects:
        print(f"Processing {test_subject}...")
        train_df = df[df['subject'] != test_subject]
        test_df = df[df['subject'] == test_subject].copy()
        test_df = test_df.sort_values('window_id')
        
        X_train = train_df[feature_cols].values
        y_train = train_df['binary_label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['binary_label'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)
        
        y_pred, rolling_acc = simulate_incremental(X_test_scaled, y_test, clf)
        results[test_subject] = rolling_acc

    rolling_acc_S10 = results['S10']
    rolling_acc_S14 = results['S14']
    rolling_acc_S17 = results['S17']
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rolling_acc_S10)), rolling_acc_S10, label='S10', linewidth=2)
    plt.plot(range(len(rolling_acc_S14)), rolling_acc_S14, label='S14', linewidth=2)
    plt.plot(range(len(rolling_acc_S17)), rolling_acc_S17, label='S17', linewidth=2)
    plt.xlabel('Window number (60s each)', fontsize=12)
    plt.ylabel('Rolling Accuracy (last 100 windows)', fontsize=12)
    plt.title('Incremental Learning Adaptation Over Shift', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves_v2.png', bbox_inches='tight')
    print("Saved plot to learning_curves_v2.png")

if __name__ == '__main__':
    main()
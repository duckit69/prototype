import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. Enter your per‑subject confusion matrices (from Tables I & II)
# Format: [ [TN, FP], [FN, TP] ]
# ============================================================

generic_cm = {
    'S2':  [[52, 7], [3, 18]],
    'S3':  [[47,15], [9,13]],
    'S4':  [[63, 0], [1,20]],
    'S5':  [[67, 2], [0,21]],
    'S6':  [[72, 3], [1,21]],
    'S7':  [[65, 4], [0,21]],
    'S8':  [[61, 0], [20,3]],
    'S9':  [[70, 0], [19,2]],
    'S10': [[72, 1], [13,11]],
    'S11': [[74, 1], [6,17]],
    'S13': [[63,14], [1,21]],
    'S14': [[62, 0], [21,0]],
    'S15': [[71, 1], [18,5]],
    'S16': [[68, 1], [12,10]],
    'S17': [[68, 0], [23,1]],
}

personalized_cm = {
    'S2':  [[58, 1], [1, 20]],
    'S3':  [[60, 2], [4, 18]],
    'S4':  [[63, 0], [0, 21]],
    'S5':  [[69, 0], [0, 21]],
    'S6':  [[74, 1], [0, 22]],
    'S7':  [[69, 0], [1, 20]],
    'S8':  [[60, 1], [1, 22]],
    'S9':  [[68, 2], [1, 20]],
    'S10': [[73, 0], [2, 22]],
    'S11': [[75, 0], [2, 21]],
    'S13': [[76, 1], [1, 21]],
    'S14': [[61, 1], [2, 19]],
    'S15': [[71, 1], [2, 21]],
    'S16': [[68, 1], [2, 20]],
    'S17': [[66, 2], [4, 20]],
}

# ============================================================
# 2. Compute average confusion matrices (absolute counts)
# ============================================================
def average_cm(cms):
    sum_cm = np.zeros((2,2))
    for cm in cms.values():
        sum_cm += np.array(cm)
    return sum_cm / len(cms)  # average absolute counts

avg_generic = average_cm(generic_cm)
avg_personalized = average_cm(personalized_cm)

# Also compute row‑normalised (percentages) for false‑negative/positive rates
def row_normalise(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    # avoid division by zero
    return np.divide(cm, row_sums, where=row_sums!=0)

avg_generic_norm = row_normalise(avg_generic)
avg_personalized_norm = row_normalise(avg_personalized)

print("Average Generic CM (absolute):\n", avg_generic)
print("Average Generic CM (row‑%):\n", avg_generic_norm)
print("Average Personalized CM (absolute):\n", avg_personalized)
print("Average Personalized CM (row‑%):\n", avg_personalized_norm)

# ============================================================
# 3. Plot heatmaps
# ============================================================
# Labels
classes = ['Non‑Stress', 'Stress']

# Use seaborn style
sns.set_theme(style='whitegrid')
plt.rcParams['font.size'] = 12

# Figure with two subplots side by side (absolute counts)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Generic heatmap (absolute)
sns.heatmap(avg_generic, annot=True, fmt='g', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Count'}, ax=axes[0])
axes[0].set_title('Generic Model')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

# Personalized heatmap (absolute)
sns.heatmap(avg_personalized, annot=True, fmt='g', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Count'}, ax=axes[1])
axes[1].set_title('Personalized Model')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()
plt.savefig('confusion_heatmaps_absolute.png', dpi=300)
plt.show()

# Optional: row‑normalised heatmaps (percentages, better for comparing error rates)
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

sns.heatmap(avg_generic_norm, annot=True, fmt='.2%', cmap='Reds',
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Percentage'}, ax=axes2[0])
axes2[0].set_title('Generic Model (Row‑Normalised)')
axes2[0].set_xlabel('Predicted')
axes2[0].set_ylabel('True')

sns.heatmap(avg_personalized_norm, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Percentage'}, ax=axes2[1])
axes2[1].set_title('Personalized Model (Row‑Normalised)')
axes2[1].set_xlabel('Predicted')
axes2[1].set_ylabel('True')

plt.tight_layout()
plt.savefig('confusion_heatmaps_normalised.png', dpi=300)
plt.show()
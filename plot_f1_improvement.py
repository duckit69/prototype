import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data
subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
generic_f1  = np.array([0.7826, 0.5200, 0.9756, 0.9545, 0.9130, 0.9130, 0.2308, 0.1739, 0.6111, 0.8293, 0.7368, 0.0000, 0.3448, 0.6061, 0.0800])
personal_f1 = np.array([0.9524, 0.8571, 1.0000, 1.0000, 0.9778, 0.9756, 0.9565, 0.9302, 0.9565, 0.9545, 0.9545, 0.9268, 0.9333, 0.9302, 0.8696])
improvements = personal_f1 - generic_f1

# Sort by improvement for better visualization
sorted_indices = np.argsort(improvements)[::-1]
sorted_subjects = [subjects[i] for i in sorted_indices]
sorted_improvements = [improvements[i] for i in sorted_indices]

# Plotting
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Create bar plot
bars = plt.bar(sorted_subjects, sorted_improvements, color=sns.color_palette("viridis", len(subjects)))

# Highlight subjects with >= 0.5 gain
# Or just let the sorted plot speak for itself
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'+{yval:.2f}', va='bottom', ha='center', fontsize=10, fontweight='bold')

plt.title('Per-Subject Improvement in F1-Score (Personalized vs. Generic)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Subject', fontsize=14)
plt.ylabel('F1-Score Absolute Improvement', fontsize=14)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='0.5 Gain Threshold')
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('f1_improvement.png', dpi=300, bbox_inches='tight')
print("Saved plot to f1_improvement.png")

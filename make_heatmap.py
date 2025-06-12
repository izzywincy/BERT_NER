import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import zoom

pre_augmented_train_eval = np.array([
    [885, 16, 0, 0, 4, 46, 66],
    [2, 88, 0, 0, 5, 0, 7],
    [4, 7, 18, 0, 0, 0, 11],
    [0, 0, 0, 125, 0, 0, 0],
    [2, 13, 0, 0, 695, 1, 16],
    [53, 0, 0, 0, 0, 1805, 53]
])

pre_augmented_test = np.array([
    [310, 0, 0, 0, 3, 15, 18],
    [3, 77, 5, 0, 4, 0, 12],
    [0, 0, 9, 0, 0, 0, 0],
    [0, 0, 0, 52, 0, 0, 0],
    [3, 0, 0, 0, 251, 1, 4],
    [7, 0, 0, 0, 0, 416, 7]
])

post_augmented_train_eval = np.array([
    [3077, 13, 1, 0, 25, 239, 278],
    [16, 368, 6, 0, 31, 12, 65], 
    [1, 13, 161, 0, 7, 2, 23],
    [0, 0, 0, 401, 0, 0, 0],
    [8, 18, 10, 0, 2894, 4, 40],
    [88, 1, 1, 0, 12, 7176, 102]
])

post_augmented_test = np.array([
    [817, 6, 0, 0, 4, 50, 60],
    [9, 172, 1, 0, 1, 3, 14],
    [0, 6, 42, 0, 8, 0, 14],
    [0, 0, 0, 161, 1, 0, 1],
    [2, 10, 1, 0, 709, 5, 18],
    [15, 0, 0, 0, 5, 1391, 20]
])

selectedMatrix = pre_augmented_test
col_labels = ["INS", "STA", "RA", "PROM_DATE", "CASE_NUM", "PERSON", "Missed"]
row_labels = ["INS", "STA", "RA", "PROM_DATE", "CASE_NUM", "PERSON"]

# Compute percentage matrix (normalized by row sum)
percentage_matrix = np.zeros_like(selectedMatrix, dtype=np.float64)
for i in range(selectedMatrix.shape[0]):
    row_sum = selectedMatrix[i].sum()
    if row_sum != 0:
        percentage_matrix[i] = (selectedMatrix[i] / row_sum) * 100  # Normalize by row sum

# Plotting the heatmap
plt.figure(figsize=(10, 6), dpi=300)

ax = sns.heatmap(
    percentage_matrix, 
    cmap='YlOrRd',
    cbar=True,
    xticklabels=col_labels,
    yticklabels=row_labels,
    annot=True,  # Annotate with percentages
    fmt='.1f',  # Format annotations as percentages with one decimal place
    annot_kws={'size': 10, 'weight': 'bold', 'color': 'black'}
)

# Customize the color bar
colorbar = ax.collections[0].colorbar
colorbar.locator = ticker.MaxNLocator(integer=True)
colorbar.update_ticks()

# Title and axis labels
plt.title('Pre-Augmented Confusion Matrix (Test) - Normalized by Row')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Adjust layout for tight fit
plt.tight_layout()

# Show and save the plot
plt.savefig("pre-aug-conf-test-normalized.png", bbox_inches='tight')
plt.show()
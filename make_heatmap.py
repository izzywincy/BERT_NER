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

# Matrix setup
selectedMatrix = pre_augmented_test
col_labels = ["INS","STA","RA","PROM_DATE","CASE_NUM","PERSON","Missed"]
row_labels = ["INS","STA","RA","PROM_DATE","CASE_NUM","PERSON"]

# Normalize each row 
# Row-wise normalize to 0–100
normalized_matrix = np.zeros_like(selectedMatrix, dtype=np.float64)
for i in range(selectedMatrix.shape[0]):
    row = selectedMatrix[i]
    row_min, row_max = row.min(), row.max()
    if row_max > row_min:
        normalized_matrix[i] = (row - row_min) / (row_max - row_min) * 100

# Set diagonal to 0 (so it appears yellow on YlOrRd)
highlight_matrix = normalized_matrix.copy().astype(np.float64)

# Step 1: zero out diagonals
for i in range(min(highlight_matrix.shape[0], highlight_matrix.shape[1])):
    highlight_matrix[i, i] = 0

# Step 2: row-wise normalize to 0–100
for i in range(highlight_matrix.shape[0]):
    row = highlight_matrix[i]
    min_val, max_val = row.min(), row.max()
    if max_val > min_val:
        highlight_matrix[i] = (row - min_val) / (max_val - min_val) * 100
    else:
        highlight_matrix[i] = 0


# Compute percentage matrix (non-raw value)
percentage_matrix = np.zeros_like(selectedMatrix, dtype=np.float64)
for i in range(selectedMatrix.shape[0]):
    row_sum = selectedMatrix[i].sum()
    if row_sum != 0:
        percentage_matrix[i] = (selectedMatrix[i] / row_sum) * 100

plt.figure(figsize=(10, 6), dpi=300)
zoom_factor = 10
smooth_matrix = zoom(highlight_matrix, zoom=zoom_factor, order=3)
smooth_matrix = np.clip(smooth_matrix, 0, 100)

ax = sns.heatmap(
    smooth_matrix, 
    cmap='YlOrRd',
    cbar=True,
    xticklabels=False,
    yticklabels=False
    )

# Overlay original values at correct zoomed-in positions
for i in range(selectedMatrix.shape[0]):
    for j in range(selectedMatrix.shape[1]):
        raw_val = selectedMatrix[i, j]
        percent_val = percentage_matrix[i, j]
        if raw_val != 0:
            text = f"{raw_val}\n({percent_val:.0f}%)"
            plt.text(
                j * zoom_factor + zoom_factor / 2,
                i * zoom_factor + zoom_factor / 2,
                text,
                ha='center',
                va='center',
                fontsize=8,
                color='black',
                weight='bold'
            )

plt.xticks(
    ticks=[j * zoom_factor + zoom_factor / 2 for j in range(selectedMatrix.shape[1])],
    labels=col_labels,  # was row_labels before
    rotation=45,
    ha='right'
)
plt.yticks(
    ticks=[i * zoom_factor + zoom_factor / 2 for i in range(selectedMatrix.shape[0])],
    labels=row_labels,  # was col_labels before
    rotation=0
)

colorbar = ax.collections[0].colorbar
colorbar.locator = ticker.MaxNLocator(integer=True)
colorbar.update_ticks()

plt.title('Pre-Augmented Confusion Matrix (Test)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("pre-aug-conf-test.png", bbox_inches='tight')
plt.show()


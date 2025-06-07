import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import zoom

col_labels = ["INS","STA","RA","PROM_DATE","CASE_NUM","PERSON","Missed"]
row_labels = ["INS","STA","RA","PROM_DATE","CASE_NUM","PERSON"]

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

# EDIT HERE
selectedMatrix = pre_augmented_test

# Normalize each row 
normalized_matrix = np.zeros_like(selectedMatrix, dtype=np.float64)
for i in range(selectedMatrix.shape[0]):
    row = selectedMatrix[i]
    min_val, max_val = row.min(), row.max()
    if max_val - min_val == 0:
        normalized_matrix[i] = 0  # Avoid division by zero
    else:
        normalized_matrix[i] = (row - min_val) / (max_val - min_val)

plt.figure(figsize=(10, 6), dpi=300)

zoom_factor = 10
smooth_matrix = zoom(normalized_matrix, zoom=zoom_factor, order=3)

sns.heatmap(
    smooth_matrix, 
    cmap='YlOrRd',
    cbar=True,
    xticklabels=False,
    yticklabels=False
    )
    
# Overlay original values at correct zoomed-in positions
for i in range(selectedMatrix.shape[0]):
    for j in range(selectedMatrix.shape[1]):
        val = selectedMatrix[i, j]
        plt.text(
            j * zoom_factor + zoom_factor / 2,
            i * zoom_factor + zoom_factor / 2,
            str(val),
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


plt.title('Pre-Augmented Confusion Matrix (Test)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("pre-aug-conf-test.png", bbox_inches='tight')
plt.show()


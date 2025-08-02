import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Define all matrices
pre_augmented_train_eval = np.array([
    [885, 16, 0, 0, 4, 46, 66],
    [2, 88, 0, 0, 5, 0, 7],
    [4, 7, 18, 0, 0, 0, 11],
    [0, 0, 0, 125, 0, 0, 0],
    [2, 13, 0, 0, 695, 1, 16],
    [53, 0, 0, 0, 0, 1805, 53]
])

pre_augmented_test = np.array([
    [319, 0, 0, 0, 3, 15, 18],
    [3, 77, 5, 0, 4, 0, 12],
    [0, 0, 9, 0, 0, 0, 0],
    [0, 0, 0, 52, 0, 0, 0],
    [3, 0, 0, 0, 251, 1, 4],
    [7, 0, 0, 0, 0, 416, 7]
])

post_augmented_train_eval = np.array([
    [3055, 13, 0, 0, 12, 219, 244],
    [17, 336, 12, 0, 16, 6, 51], 
    [1, 41, 128, 0, 0, 0, 42],
    [0, 0, 0, 384, 0, 0, 0],
    [10, 16, 6, 0, 2857, 13, 45],
    [114, 2, 0, 0, 0, 7178, 116]
])

post_augmented_test = np.array([
    [827, 14, 0, 0, 5, 46, 65],
    [4, 157, 7, 0, 6, 4, 21],
    [0, 2, 43, 0, 8, 0, 10],
    [0, 0, 0, 164, 1, 0, 1],
    [1, 0, 3, 0, 729, 7, 11],
    [12, 1, 0, 0, 6, 1379, 19]
])

# Store all matrices with their labels
matrices = {
    'Pre-Augmented w/o CNS (Train-Eval)': pre_augmented_train_eval,
    'Pre-Augmented w/o CNS (Test)': pre_augmented_test,
    'Post-Augmented w/o CNS (Train-Eval)': post_augmented_train_eval,
    'Post-Augmented w/o CNS (Test)': post_augmented_test
}

# Define labels
col_labels = ["INS", "CNS", "STA", "RA", "PROM_DATE", "CASE_NUM", "PERSON", "Missed"]
row_labels = ["INS", "CNS", "STA", "RA", "PROM_DATE", "CASE_NUM", "PERSON"]

# Process and plot each matrix
for title, selectedMatrix in matrices.items():
    # Normalize by row to get percentage matrix
    percentage_matrix = np.zeros_like(selectedMatrix, dtype=np.float64)
    for i in range(selectedMatrix.shape[0]):
        row_sum = selectedMatrix[i].sum()
        if row_sum != 0:
            percentage_matrix[i] = (selectedMatrix[i] / row_sum) * 100

    # Plot setup
    plt.figure(figsize=(10, 6), dpi=300)
    cmap = plt.get_cmap('YlGn')
    norm = Normalize(vmin=percentage_matrix.min(), vmax=percentage_matrix.max())

    ax = sns.heatmap(
        percentage_matrix,
        cmap=cmap,
        cbar=True,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=False
    )

    # Annotate each cell with dynamic text color
    for i in range(percentage_matrix.shape[0]):
        for j in range(percentage_matrix.shape[1]):
            value = percentage_matrix[i, j]
            rgba = cmap(norm(value))
            r, g, b, _ = rgba
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = 'white' if brightness < 0.5 else 'black'

            ax.text(
                j + 0.5, i + 0.5,
                f"{value:.1f}%",
                ha='center', va='center',
                color=text_color,
                fontsize=10,
                weight='bold'
            )

    # Colorbar customization
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 100])

    # Move x-axis labels to top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_tick_params(labeltop=True, labelbottom=False)
    ax.set_xlabel('Predicted Label', labelpad=20)
    ax.xaxis.label.set_position((0.5, 1.1))
    plt.title(f'Classification Matrix - {title}')
    plt.ylabel('True Label')

    # Layout and saving
    plt.tight_layout()
    save_path = f'./plots/{title.replace(" ", "_").lower()}_matrix.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"Plot for {title} saved to: {save_path}")

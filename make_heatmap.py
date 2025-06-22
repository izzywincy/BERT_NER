import os
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

# Example matrices stored in a dictionary
matrices = {
    'Pre-Augmented (Train-Eval)': pre_augmented_train_eval,
    'Pre-Augmented (Test)': pre_augmented_test,
    'Post-Augmented (Train-Eval)': post_augmented_train_eval,
    'Post-Augmented (Test)': post_augmented_test
}

col_labels = ["INS", "STA", "RA", "PROM_DATE", "CASE_NUM", "PERSON", "Missed"]
row_labels = ["INS", "STA", "RA", "PROM_DATE", "CASE_NUM", "PERSON"]

# Loop through each matrix in the dictionary
for title, selectedMatrix in matrices.items():
    # Compute percentage matrix (normalized by row sum)
    percentage_matrix = np.zeros_like(selectedMatrix, dtype=np.float64)
    for i in range(selectedMatrix.shape[0]):
        row_sum = selectedMatrix[i].sum()
        if row_sum != 0:
            percentage_matrix[i] = (selectedMatrix[i] / row_sum) * 100  # Normalize by row sum

    # Plotting the heatmap without formatting the annotations
    plt.figure(figsize=(10, 6), dpi=300)

    # Create the heatmap without 'fmt' or 'annot_kws'
    ax = sns.heatmap(
        percentage_matrix, 
        cmap='YlOrRd',
        cbar=True,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=False  # Set annot=False to avoid internal formatting issues
    )

    # Manually annotate the cells with formatted percentages
    for i in range(percentage_matrix.shape[0]):
        for j in range(percentage_matrix.shape[1]):
            ax.text(
                j + 0.5, i + 0.5,  # Position the text (adjust for centering)
                f"{percentage_matrix[i, j]:.1f}%",  # Format the value with the "%" sign
                ha='center', va='center',  # Align the text in the center
                color='black', fontsize=10, weight='bold'  # Text formatting
            )

    # Customize the color bar to show only 0 and 100
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 100])  # Set only 0 and 100 ticks on the color bar

    # Move x-axis labels to the top and ensure they are visible
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_tick_params(labeltop=True, labelbottom=False)  # Remove labels at the bottom

    # Set x-axis label (Predicted Label) above the x-axis labels
    ax.set_xlabel('Predicted Label', labelpad=20)  # Add xlabel with padding to position it higher
    ax.xaxis.label.set_position((0.5, 1.1))  # Position the xlabel above the x-axis labels

    # Title and axis labels
    plt.title(f'Classification Matrix - {title}')
    plt.ylabel('True Label')

    # Adjust layout for tight fit
    plt.tight_layout()

    # Define the path and ensure the directory exists (use relative path for current directory)
    save_path = f'./plots/{title.replace(" ", "_").lower()}_matrix.png'  # Save with dynamic name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist

    # Show and save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    # Inform the user where the plot was saved
    print(f"Plot for {title} saved to: {save_path}")
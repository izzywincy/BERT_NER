import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import os

def plot_percentage_heatmap(percentage_matrix, row_labels, col_labels, title, save_path=None):
    """
    Plot a heatmap using an already normalized percentage matrix.

    Parameters:
    - percentage_matrix: 2D NumPy array with values from 0 to 100
    - row_labels: list of strings for Y-axis
    - col_labels: list of strings for X-axis
    - title: string for the plot title
    - save_path: optional string, path to save the PNG image
    """
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

    # Colorbar tweaks
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 100])

    # Move x-axis labels to top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_tick_params(labeltop=True, labelbottom=False)
    ax.set_xlabel('Predicted Label', labelpad=20)
    ax.xaxis.label.set_position((0.5, 1.1))

    plt.title(f'Classification Matrix - {title}')
    plt.ylabel('True Label')
    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    plt.show()

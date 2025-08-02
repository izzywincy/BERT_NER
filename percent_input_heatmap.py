import numpy as np
from make_heatmap_percent import plot_percentage_heatmap  # replace with actual import

# Matrix values from the heatmap (in %)
# Step 1: Define your row and column labels once
row_labels = ["INS", "STA", "RA", "PROM_DATE", "CASE_NUM", "PERSON"]
col_labels = ["INS", "STA", "RA", "PROM_DATE", "CASE_NUM", "PERSON", "Missed"]

# Step 2: Create a list of dictionaries, each containing matrix data and metadata
matrices = [
    {
        "matrix": np.array([
            [87.0, 1.6, 0.0, 0.0, 0.4, 4.5, 6.5],
            [2.0, 86.3, 0.0, 0.0, 4.9, 0.0, 6.9],
            [10.0, 17.5, 45.0, 0.0, 0.0, 0.0, 27.5],
            [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
            [0.3, 1.8, 0.0, 0.0, 95.6, 0.1, 2.2],
            [2.8, 0.0, 0.0, 0.0, 0.0, 94.5, 2.8]
        ]),
        "title": "Pre-Augmented (Train-Eval)",
        "save_path": "./plots/pre_augmented_train_eval.png"
    },
    {
        "matrix": np.array([
            [89.9, 0.0, 0.0, 0.0, 0.8, 4.2, 5.1],
            [3.0, 76.2, 5.0, 0.0, 4.0, 0.0, 11.9],
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0, 0.0, 96.9, 0.4, 1.5],
            [1.6, 0.0, 0.0, 0.0, 0.0, 96.7, 1.6]
        ]),
        "title": "Pre-Augmented (Test)",
        "save_path": "./plots/pre_augmented_test.png"
    },
    # Add 3 more entries below
    {
        "matrix": ...,  # replace with your 3rd matrix
        "title": "Post-Augmented (Train-Eval)",
        "save_path": "./plots/post_augmented_train_eval.png"
    },
    {
        "matrix": ...,  # replace with your 4th matrix
        "title": "Post-Augmented (Test)",
        "save_path": "./plots/post_augmented_test.png"
    },
    {
        "matrix": ...,  # replace with your 5th matrix
        "title": "Final Evaluation Set",
        "save_path": "./plots/final_eval_set.png"
    }
]

for entry in matrices:
    plot_percentage_heatmap(
        percentage_matrix=entry["matrix"],
        row_labels=row_labels,
        col_labels=col_labels,
        title=entry["title"],
        save_path=entry["save_path"]
    )

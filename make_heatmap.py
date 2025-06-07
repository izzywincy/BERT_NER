import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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

plt.figure(figsize=(8, 6), dpi=300)

sns.heatmap(
    pre_augmented_test, 
    annot=True, 
    fmt='d', 
    cmap='YlOrRd', 
    linewidths=0.5,
    cbar=True,
    vmin=0, # minimum value for color scale
    vmax=100 # maximum value for color scale
    )
plt.title('Pre-Augmented Confusion Matrix (Train/Eval)')

plt.savefig("pre-aug-conf-train-eval.png", bbox_inches='tight')

plt.show()


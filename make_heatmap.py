import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

matrix = np.array([
    [885, 16, 0, 0, 4, 46, 66],
    [2, 88, 0, 0, 5, 0, 7],
    [4, 7, 18, 0, 0, 0, 11],
    [0, 0, 0, 125, 0, 0, 0],
    [2, 13, 0, 0, 695, 1, 16],
    [53, 0, 0, 0, 0, 1805, 53]
])

plt.figure(figsize=(8, 6), dpi=300)

sns.heatmap(
    matrix, 
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


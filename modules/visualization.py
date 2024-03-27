""" Contains visualization functionalities """

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

DPI = 96


def vis_confusion_matrix(confusion_matrix: np.ndarray, labels_x: list, labels_y: list, out_path: str, size_px: tuple = (1920, 960)) -> None:
    plt.figure(figsize=(int(size_px[0] / DPI), int(size_px[1] / DPI)))
    ax = sns.heatmap(
        confusion_matrix,
        annot=False,
        cmap='RdYlGn',
        linewidths=0.5,
        xticklabels=labels_x,
        yticklabels=labels_y,
        square=True,
    )
    plt.xlabel('Predicted class', fontsize=20)
    plt.ylabel('Real class', fontsize=20)
    plt.title('Confusion Matrix')
    plt.savefig(fname=out_path, transparent=False)

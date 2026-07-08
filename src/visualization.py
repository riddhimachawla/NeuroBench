import os
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self):

        os.makedirs("results/figures", exist_ok=True)

    def plot_confusion_matrix(self, cm, class_names):

        plt.figure(figsize=(7, 6))

        plt.imshow(cm, interpolation="nearest", cmap="Blues")

        plt.title("Confusion Matrix")

        plt.colorbar()

        tick_marks = np.arange(len(class_names))

        plt.xticks(tick_marks, class_names)

        plt.yticks(tick_marks, class_names)

        threshold = cm.max() / 2

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):

                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    color="white" if cm[i, j] > threshold else "black",
                )

        plt.ylabel("True Label")

        plt.xlabel("Predicted Label")

        plt.tight_layout()

        plt.savefig(
            "results/figures/confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    def plot_accuracy(self, results_df):

        plt.figure(figsize=(9, 5))

        plt.bar(
            results_df["Model"],
            results_df["Accuracy"]
        )

        plt.ylabel("Accuracy")

        plt.title("Model Accuracy Comparison")

        plt.xticks(rotation=20)

        plt.tight_layout()

        plt.savefig(
            "results/figures/model_accuracy.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    def plot_cross_validation(self, results_df):

        plt.figure(figsize=(9,5))

        plt.bar(

            results_df["Model"],

            results_df["Cross Validation"]

        )

        plt.ylabel("Cross Validation Accuracy")

        plt.title("Cross Validation Comparison")

        plt.xticks(rotation=20)

        plt.tight_layout()

        plt.savefig(

            "results/figures/cross_validation.png",

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()
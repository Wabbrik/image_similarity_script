import os
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

assets = sorted(
    os.listdir("assets"),
    key=lambda name: int(name.split(".")[0]),
)


def segments(image: ArrayLike, lon: int, lat: int, apply: Callable[[ArrayLike], ArrayLike]) -> ArrayLike:
    h, w = np.shape(image)
    seg_h, seg_w = (h // lon), (w // lon)

    return np.array(
        sorted(
            [
                apply(image[(row * seg_h) : ((row + 1) * seg_h), (col * seg_w) : ((col + 1) * seg_w)])
                for row in range(lat)
                for col in range(lon)
            ]
        )
    )


def distance(x: ArrayLike, y: ArrayLike) -> float:
    return np.linalg.norm(x - y)


def draw_heatmap(heatmap: ArrayLike, labels_x: list[str], labels_y: list[str]) -> None:
    fig, ax = plt.subplots()
    ax.imshow(heatmap)

    ax.set_xticks(np.arange(len(labels_x)), labels=labels_x)
    ax.set_yticks(np.arange(len(labels_y)), labels=labels_y)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(labels_x)):
        for j in range(len(labels_y)):
            ax.text(j, i, f"{heatmap[i, j]:.1f}", ha="center", va="center", color="w", rotation=45)

    fig.tight_layout()
    plt.show()


def main():
    images = [cv2.imread(os.path.join("assets", f), flags=0) for f in assets]

    scores = [segments(image, lon=8, lat=8, apply=lambda segment: np.mean(segment)) for image in images]

    distances = np.array([distance(score1, score2) for score1 in scores for score2 in scores]).reshape(
        len(scores), len(scores)
    )

    draw_heatmap(distances, assets, assets)


if __name__ == "__main__":
    main()

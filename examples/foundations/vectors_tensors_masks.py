"""Small NumPy examples for scalar, vector, matrix, tensor, embedding and mask concepts."""
from __future__ import annotations

import numpy as np


def main() -> None:
    scalar = np.array(3.5)
    vector = np.array([0.2, 0.8, -0.1])
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    tensor = np.arange(24).reshape(2, 3, 4)
    embedding = np.array([0.15, -0.22, 0.91, 0.08])
    mask = matrix > 3

    print("scalar shape:", scalar.shape)
    print("vector norm:", np.linalg.norm(vector))
    print("matrix transpose:\n", matrix.T)
    print("tensor shape:", tensor.shape)
    print("embedding dimension:", embedding.size)
    print("masked values:", matrix[mask])


if __name__ == "__main__":
    main()

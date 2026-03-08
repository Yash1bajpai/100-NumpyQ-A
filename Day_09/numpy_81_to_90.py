# ==========================================
# Day 9: NumPy Questions 81 - 90
# ==========================================
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# 81. Generate sliding windows of size 4 from an array
Z_seq = np.arange(1, 15)
R = sliding_window_view(Z_seq, window_shape=4)
print("81. Sliding windows of size 4:\n", R[:3])

# 82. Compute a matrix rank
Z_rank = np.random.uniform(0, 1, (10, 10))
rank = np.linalg.matrix_rank(Z_rank)
print("\n82. Matrix Rank:", rank)

# 83. Find the most frequent value in an array
Z_freq = np.random.randint(0, 10, 50)
most_frequent = np.bincount(Z_freq).argmax()
print("\n83. Most frequent value:", most_frequent)

# 84. Extract all contiguous 3x3 blocks from a 10x10 matrix
Z_blocks = np.random.randint(0, 5, (10, 10))
blocks = sliding_window_view(Z_blocks, window_shape=(3, 3))
print("\n84. Extracted 3x3 blocks shape (Convolutional view):", blocks.shape)

# 85. Create a 2D array subclass such that Z[i,j] == Z[j,i]


class Symmetric(np.ndarray):
    def __setitem__(self, index, value):
        i, j = index
        super(Symmetric, self).__setitem__((i, j), value)
        super(Symmetric, self).__setitem__((j, i), value)  # Mirror copy


def symmetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symmetric)


S = symmetric(np.zeros((3, 3)))
S[0, 1] = 42
print("\n85. Symmetric Array Subclass:\n", S)

# 86. Compute the sum of p matrix products at once using einsum
p, n = 10, 20
M = np.ones((p, n, n))
V = np.ones((p, n, 1))
result = np.einsum('pij,pjk->pik', M, V).sum(axis=0)
print("\n86. Batched matrix product sum shape:", result.shape)

# 87. Get the block-sum (4x4 blocks) of a 16x16 array
Z_grid = np.ones((16, 16))
block_sum = Z_grid.reshape(4, 4, 4, 4).sum(axis=(1, 3))
print("\n87. 4x4 Block sum shape (Pooling operation):", block_sum.shape)

# 88. Implement the Game of Life using numpy arrays


def iterate(Z):
    N = np.zeros(Z.shape, dtype=int)
    # Count neighbors using shifted slices
    N[1:-1, 1:-1] += (Z[:-2, :-2] + Z[:-2, 1:-1] + Z[:-2, 2:] +
                      Z[1:-1, :-2]                + Z[1:-1, 2:] +
                      Z[2:, :-2]  + Z[2:, 1:-1]  + Z[2:, 2:])
    birth = (N == 3) & (Z == 0)
    survive = ((N == 2) | (N == 3)) & (Z == 1)
    Z[...] = 0
    Z[birth | survive] = 1
    return Z


Z_life = np.random.randint(0, 2, (5, 5))
print("\n88. Game of Life (Initial State):\n", Z_life)
for _ in range(3):
    Z_life = iterate(Z_life)
print("    Game of Life (After 3 iterations):\n", Z_life)

# 89. Get the n largest values of an array
Z_large = np.arange(10000)
np.random.shuffle(Z_large)
n_top = 5
# argpartition runs in linear time O(N), much faster than sorting O(N log N)
top_n_indexes = np.argpartition(Z_large, -n_top)[-n_top:]
print(f"\n89. Top {n_top} values (Top-K sampling logic):", Z_large[top_n_indexes])

# 90. Build the cartesian product of an arbitrary number of vectors
arrays = ([1, 2, 3], [4, 5], [6, 7])
grid = np.meshgrid(*arrays, indexing='ij')
cartesian = np.stack(grid, axis=-1).reshape(-1, len(arrays))
print("\n90. Cartesian Product (First 5 combinations):\n", cartesian[:5])

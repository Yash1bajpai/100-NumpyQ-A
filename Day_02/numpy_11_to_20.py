# ==========================================
# Day 2: NumPy Questions 11 - 20
# ==========================================
import numpy as np

# 11. Create a 3x3 identity matrix
arr_identity = np.eye(3)
print("11. Identity Matrix:\n", arr_identity)

# 12. Create a 3x3x3 array with random values
arr_3d = np.random.random((3, 3, 3))
print("\n12. 3x3x3 Random Array shape:", arr_3d.shape)

# 13. Create a 10x10 array with random values and find min/max
arr_10x10 = np.random.random((10, 10))
print(f"\n13. Min: {arr_10x10.min():.4f}, Max: {arr_10x10.max():.4f}")

# 14. Create a random vector of size 30 and find the mean value
arr_vec = np.random.random(30)
print(f"\n14. Mean of 30 random values: {arr_vec.mean():.4f}")

# 15. Create a 2d array with 1 on the border and 0 inside
arr_border = np.ones((5, 5))
arr_border[1:-1, 1:-1] = 0
print("\n15. Border 1, Inside 0:\n", arr_border)

# 16. Add a border (filled with 0's) around an existing array
arr_pad = np.ones((3, 3))
arr_pad = np.pad(arr_pad, pad_width=1, mode='constant', constant_values=0)
print("\n16. Padded array:\n", arr_pad)

# 17. What is the result of the following expression?
print("\n17. Nan and Float evaluations:")
print("0 * np.nan:", 0 * np.nan)
print("np.nan == np.nan:", np.nan == np.nan)
print("np.inf > np.nan:", np.inf > np.nan)
print("np.nan - np.nan:", np.nan - np.nan)
print("0.3 == 3 * 0.1:", 0.3 == 3 * 0.1)
print("Using np.isclose(0.3, 3 * 0.1):", np.isclose(0.3, 3 * 0.1))  # The ML way

# 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
arr_diag = np.diag([1, 2, 3, 4], k=-1)
print("\n18. Values below diagonal:\n", arr_diag)

# 19. Create an 8x8 matrix and fill it with a checkerboard pattern
arr_checker = np.ones((8, 8))
arr_checker[1::2, ::2] = 0
arr_checker[::2, 1::2] = 0
print("\n19. Checkerboard:\n", arr_checker)

# 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
index_100 = np.unravel_index(99, (6, 7, 8))
print("\n20. Index of 100th element:", index_100)

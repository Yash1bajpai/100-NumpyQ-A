# ==========================================
# Day 3: NumPy Questions 21 - 30
# ==========================================
import numpy as np
import warnings

# 21. Create a checkerboard 8x8 matrix using the tile function
arr_tile = np.tile([[0, 1], [1, 0]], reps=(4, 4))
print("21. Checkerboard using tile:\n", arr_tile)

# 22. Normalize a 5x5 random matrix (Min-Max and Standardization)
arr_rand = np.random.random((5, 5))
mi, mx = arr_rand.min(), arr_rand.max()
arr_minmax = (arr_rand - mi) / (mx - mi)  # Min-Max Scaling
print("\n22. Min-Max Normalization:\n", arr_minmax)

Z = np.random.normal(loc=150, scale=10, size=(5, 5))
Z_standardized = (Z - np.mean(Z)) / np.std(Z)  # Z-Score Standardization
print("Z-Score Standardization:\n", Z_standardized)

# 23. Create custom dtype that describes color as four unsigned bytes (RGBA)
color_dtype = np.dtype([('r', np.ubyte), ('g', np.ubyte), ('b', np.ubyte), ('a', np.ubyte)])
print("\n23. Custom RGBA dtype:", color_dtype)

# 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
m1 = np.random.randint(1, 10, size=(5, 3))
m2 = np.random.randint(1, 10, size=(3, 2))
result = m1 @ m2  # Modern Python matrix multiplication
print("\n24. Matrix Multiplication (5x3 @ 3x2):\n", result)

# 25. Given a 1D array, negate all elements which are between 3 and 8, in place.
arr_negate = np.arange(1, 11)
arr_negate[(3 < arr_negate) & (arr_negate < 8)] *= -1
print("\n25. Negated elements between 3 and 8:", arr_negate)

# 26. What is the output of the following script?
print("\n26. sum(range(5), -1) Output:", sum(range(5), -1))

# 27. Consider an integer vector Z, which of these expressions are legal?
print("\n27. Z**Z, 2 << Z >> 2, Z <- Z, 1j*Z, Z/1/1 are all legal.")
print("Z < Z > Z is illegal (ValueError: ambiguous truth value).")

# 28. What are the result of the following expressions?
warnings.filterwarnings('ignore')  # Ignoring the divide by zero warning for clean output
print("\n28. np.array(0) / np.array(0):", np.array(0) / np.array(0))
print("np.array(0) // np.array(0):", np.array(0) // np.array(0))
print("np.array([np.nan]).astype(int).astype(float):", np.array([np.nan]).astype(int).astype(float))

# 29. How to round away from zero a float array ?
arr_float = np.array([1.2, 3.4, 5.5, -1.2, -5.5, 0.5, 0])
rounded_away = np.where(arr_float > 0, np.ceil(arr_float), np.floor(arr_float))
print("\n29. Rounded away from zero:", rounded_away)

# 30. How to find common values between two arrays?
arr_a = np.random.randint(-10, 10, 10)
arr_b = np.random.randint(-10, 10, 10)
print("\n30. Common elements:", np.intersect1d(arr_a, arr_b))

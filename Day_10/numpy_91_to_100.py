# ==========================================
# Day 10: NumPy Questions 91 - 100
# ==========================================
import numpy as np
import time

# 91. Create a record array from a regular array
Z_91 = np.array([("Hello", 2.5, 3), ("World", 3.6, 2)])
R_91 = np.core.records.fromarrays(Z_91.T, names='col1, col2, col3', formats='S8, f8, i8')
print("91. Record Array col1:", R_91.col1)

# 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods
x_92 = np.random.rand(int(1e6))
print("\n92. Speed Test for Z^3 (1 Million elements):")

start = time.perf_counter()
np.power(x_92, 3)
print(f"    np.power time:  {time.perf_counter() - start:.5f} sec")

start = time.perf_counter()
_ = x_92 * x_92 * x_92
print(f"    x * x * x time: {time.perf_counter() - start:.5f} sec (Fastest due to zero overhead)")

start = time.perf_counter()
np.einsum('i,i,i->i', x_92, x_92, x_92)
print(f"    einsum time:    {time.perf_counter() - start:.5f} sec")

# 93. Consider two arrays A and B of shape (8,3) and (2,2). Find rows of A containing elements of each row of B
A_93 = np.random.randint(0, 5, (8, 3))
B_93 = np.random.randint(0, 5, (2, 2))
C_93 = (A_93[..., np.newaxis, np.newaxis] == B_93)
rows_93 = np.where(C_93.any((3, 1)).all(1))[0]
print("\n93. Matching Rows in A:\n", rows_93)

# 94. Extract rows with unequal values
Z_94 = np.random.randint(0, 2, (5, 3))
unequal_rows_94 = Z_94[Z_94.max(axis=1) != Z_94.min(axis=1)]
print("\n94. Rows with unequal values:\n", unequal_rows_94)

# 95. Convert a numpy array of integers into a binary matrix
I_95 = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
binary_matrix_95 = ((I_95.reshape(-1, 1) & (2**np.arange(8))) != 0).astype(int)
print("\n95. Binary representation (Quantization Logic):\n", binary_matrix_95[:, ::-1])

# 96. Extract unique rows from a 2D array
Z_96 = np.random.randint(0, 2, (6, 3))
unique_rows_96 = np.unique(Z_96, axis=0)
print("\n96. Unique Rows:\n", unique_rows_96)

# 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function
A_97 = np.array([1, 2, 3])
B_97 = np.array([4, 5, 6])
print("\n97. einsum Equivalents:")
print("    Sum (A):       ", np.einsum('i->', A_97))
print("    Element-wise:  ", np.einsum('i,i->i', A_97, B_97))
print("    Inner Product: ", np.einsum('i,i->', A_97, B_97))
print("    Outer Product:\n", np.einsum('i,j->ij', A_97, B_97))

# 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples?
phi = np.arange(0, 10*np.pi, 0.1)
x_98 = phi * np.cos(phi)
y_98 = phi * np.sin(phi)
dr = (np.diff(x_98)**2 + np.diff(y_98)**2)**.5 
r = np.zeros_like(x_98)
r[1:] = np.cumsum(dr)
r_int = np.linspace(0, r.max(), 5)
x_int = np.interp(r_int, r, x_98)
print("\n98. Equidistant Sampled X coordinates:", np.round(x_int, 2))

# 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution
X_99 = np.asarray([[1.0, 0.0, 3.0, 8.0], [2.0, 0.0, 1.0, 1.0], [1.5, 2.5, 1.0, 0.0]])
n_99 = 4
mask_99 = np.logical_and.reduce(np.mod(X_99, 1) == 0, axis=-1)
mask_99 &= (X_99.sum(axis=-1) == n_99)
print("\n99. Valid multinomial rows (n=4):\n", X_99[mask_99])

# 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X
X_100 = np.random.randn(100) 
N_100 = 1000 
idx_100 = np.random.randint(0, X_100.size, (N_100, X_100.size))
means_100 = X_100[idx_100].mean(axis=1)
confint_100 = np.percentile(means_100, [2.5, 97.5])
print(f"\n100. Bootstrapped 95% Confidence Interval: {np.round(confint_100, 4)}")

# ==========================================
# Day 5: NumPy Questions 41 - 50
# ==========================================
import numpy as np
import sys

# 41. How to sum a small array faster than np.sum?
arr_small = np.arange(10)
result_reduce = np.add.reduce(arr_small)
print("41. Faster sum using add.reduce:", result_reduce)
# Note: np.add.reduce bypasses the overhead of np.sum() (data type checks, axis resolution)

# 42. Consider two random arrays A and B, check if they are equal
arr1 = np.random.randint(-10, 10, 10)
arr2 = np.random.randint(-10, 10, 10)
print("\n42. Arrays equal (integers):", np.array_equal(arr1, arr2))
print("    Arrays equal (floats safer method):", np.allclose(arr1, arr1))  # Example of allclose

# 43. Make an array immutable (read-only)
arr_immutable = np.random.randint(5, 10, 5)
arr_immutable.flags.writeable = False
print("\n43. Array made read-only. (Attempting to modify will throw ValueError)")

# 44. Consider a random 10x2 matrix representing cartesian coordinates, convert to polar
cartesian = np.random.randint(0, 25, size=(5, 2))
X, Y = cartesian[:, 0], cartesian[:, 1]
R = np.hypot(X, Y)
T = np.arctan2(Y, X)
polar_coords = np.column_stack((R, T))
print("\n44. Polar Coordinates (R, Theta):\n", polar_coords)

# 45. Create random vector of size 10 and replace the maximum value by 0
arr_max = np.random.random(10)
arr_max[arr_max.argmax()] = 0
print("\n45. Maximum value replaced by 0:\n", arr_max)

# 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area
Z_struct = np.zeros((5, 5), [('x', float), ('y', float)])
Z_struct['x'], Z_struct['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
print("\n46. Structured coordinate array:\n", Z_struct)

# 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))
X_c = np.arange(8)
Y_c = X_c + 0.5
C = 1.0 / np.subtract.outer(X_c, Y_c)
print("\n47. Determinant of Cauchy matrix:", np.linalg.det(C))

# 48. Print the minimum and maximum representable values for each numpy scalar type
print("\n48. --- Hardware Limits ---")
for dtype in [np.int8, np.int32, np.float32]:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    else:
        info = np.finfo(dtype)
    print(f"    {dtype.__name__}: Min {info.min}, Max {info.max}")

# 49. How to print all the values of an array?
# Note: Resetting it immediately after so it doesn't flood the terminal
np.set_printoptions(threshold=sys.maxsize)
arr_large = np.zeros((2, 2))  # Kept small here to avoid terminal flood, but logic applies
print("\n49. Print options threshold set to max.")
np.set_printoptions(threshold=1000)  # Reset to default

# 50. How to find the closest value (to a given scalar) in a vector?
Z_vec = np.arange(100)
v = np.random.uniform(0, 100)
index_closest = (np.abs(Z_vec - v)).argmin()
print(f"\n50. Scalar: {v:.2f} | Closest value in array: {Z_vec[index_closest]}")

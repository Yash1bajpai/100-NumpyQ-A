# ==========================================
# Day 6: NumPy Questions 51 - 60
# ==========================================
import numpy as np
import scipy.spatial
from io import StringIO

# 51. Create a structured array representing a position (x,y) and a color (r,g,b)
particle_blueprint = [('x', float), ('y', float), ('r', np.uint8), ('g', np.uint8), ('b', np.uint8)]
particles = np.zeros(5, dtype=particle_blueprint)
particles[0] = (10.5, 20.2, 255, 0, 0)
print("51. Structured Array (Position & Color):\n", particles[0])

# 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances
Z_coords = np.random.random((100, 2))
D = scipy.spatial.distance.cdist(Z_coords, Z_coords)
print("\n52. Point-to-Point Distance Matrix Shape:", D.shape)

# 53. How to convert a float (32 bits) array into an integer (32 bits) array in place?
Z_float = (np.random.rand(10) * 100).astype(np.float32)
print("\n53. Original Float32 Array:\n", Z_float)
Y_int = Z_float.view(np.int32)
Y_int[:] = Z_float  # Zero-copy, in-place cast
print("    In-place converted Int32 Array:\n", Y_int)

# 54. How to read the following file? (Filled blanks with NaN)
file_data = StringIO("""
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
""")
Z_csv = np.genfromtxt(file_data, delimiter=",", filling_values=np.nan)
print("\n54. Parsed CSV Data with NaNs:\n", Z_csv)

# 55. What is the equivalent of enumerate for numpy arrays?
Z_grid = np.arange(4).reshape(2, 2)  # Kept small for output readability
print("\n55. Equivalent of enumerate (ndenumerate):")
for index, value in np.ndenumerate(Z_grid):
    print(f"    Coordinate {index} holds value: {value}")

# 56. Generate a generic 2D Gaussian-like array
X, Y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
D_gauss = np.sqrt(X*X + Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-((D_gauss-mu)**2 / (2.0 * sigma**2)))
print("\n56. Generic 2D Gaussian Array:\n", np.round(G, 3))

# 57. How to randomly place p elements in a 2D array?
arr_p = np.zeros((5, 5))
p = 3
indices = np.random.choice(arr_p.size, p, replace=False)
np.put(arr_p, indices, 100)
print(f"\n57. Randomly placed {p} elements (value 100):\n", arr_p)

# 58. Subtract the mean of each row of a matrix
Z_matrix = np.random.rand(3, 4)
row_means = Z_matrix.mean(axis=1, keepdims=True)  # keepdims is vital here
Z_centered = Z_matrix - row_means
print("\n58. Matrix with Row Means Subtracted:\n", np.round(Z_centered, 2))

# 59. How to sort an array by the nth column?
Z_sort = np.random.randint(0, 10, (3, 3))
n = 1
sorted_Z = Z_sort[Z_sort[:, n].argsort()]
print(f"\n59. Array sorted by column index {n}:\n", sorted_Z)

# 60. How to tell if a given 2D array has null columns?
Z_nulls = np.random.randint(0, 3, (3, 5))
Z_nulls[:, 2] = 0  # Force a null column
has_null = (Z_nulls == 0).all(axis=0).any()
print("\n60. Does the array have a fully null column?", has_null)

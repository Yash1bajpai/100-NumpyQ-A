# ==========================================
# Day 7: NumPy Questions 61 - 70
# ==========================================
import numpy as np

# 61. Find the nearest value from a given value in an array
Z_near = np.random.uniform(0, 10, (3, 3))
target = 5.0
flat_index = np.abs(Z_near - target).argmin()
print(f"61. Target: {target} | Closest value: {Z_near.flat[flat_index]:.4f}")

# 62. Considering two arrays with shape (1,3) and (3,1), compute their sum using an iterator
A = np.arange(3).reshape(1, 3)
B = np.arange(3).reshape(3, 1)
iterator = np.nditer([A, B, None])
for x, y, z in iterator:
    z[...] = x + y
print("\n62. Iterator Result (Broadcasting 1x3 + 3x1):\n", iterator.operands[2])

# 63. Create an array class that has a name attribute


class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', "no name")


Z_named = NamedArray(np.arange(5), name="My Little Array")
print("\n63. Custom Array Class:")
print(f"    Original: {Z_named} | Name: {Z_named.name}")
print(f"    Sliced:   {Z_named[1:4]} | Name: {Z_named[1:4].name}")

# 64. Add 1 to each element indexed by a second vector (careful with repeated indices)
arr_add = np.zeros(5)
indices = [1, 1, 2]
# bincount is highly optimized and handles repeated indices faster than np.add.at
arr_add += np.bincount(indices, minlength=len(arr_add))
print("\n64. Array after accumulating repeated indices:", arr_add)

# 65. Accumulate elements of a vector (X) to an array (F) based on an index list (I)
X_weights = [1, 2, 3, 4, 5, 6]
I_indices = [1, 3, 9, 3, 4, 1]
F_accumulated = np.bincount(I_indices, weights=X_weights)
print("\n65. Accumulated array using bincount weights:\n", F_accumulated)

# 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors
w, h = 16, 16
image = np.random.randint(0, 4, (w, h, 3)).astype(np.ubyte)
pixels = image.reshape(-1, 3)  # Flatten to a list of RGB pixels
unique_colors = len(np.unique(pixels, axis=0))
print(f"\n66. Unique colors in {w}x{h} image: {unique_colors}")

# 67. Considering a four dimensions array, how to get sum over the last two axis at once?
Z_4d = np.random.randint(0, 10, size=(3, 4, 3, 4))
sum_last_two = Z_4d.sum(axis=(-2, -1))
print("\n67. Sum over last two axes shape:", sum_last_two.shape)

# 68. Compute means of subsets of a vector D using a vector S describing subset indices
D = np.array([80.0, 90.0, 70.0, 100.0, 60.0])
S = np.array([0, 1, 0, 1, 1])
means = np.bincount(S, weights=D) / np.bincount(S)
print("\n68. Grouped means using bincount:", means)

# 69. How to get the diagonal of a dot product?
A_mat = np.random.randint(1, 10, (3, 3))
B_mat = np.random.randint(1, 10, (3, 3))
# Method 1: Math Hack
fast_diagonal = np.sum(A_mat * B_mat.T, axis=1)
# Method 2: Einsum (Deep Learning Standard)
einsum_diagonal = np.einsum('ij,ji->i', A_mat, B_mat)
print("\n69. Diagonal of Dot Product:")
print("    Using Sum Hack:  ", fast_diagonal)
print("    Using np.einsum: ", einsum_diagonal)

# 70. Build a new vector with 3 consecutive zeros interleaved between each value
Z_orig = np.array([1, 2, 3, 4, 5])
nz = 3
Z_interleaved = np.zeros(len(Z_orig) + (len(Z_orig) - 1) * nz)
Z_interleaved[::nz+1] = Z_orig
print("\n70. Interleaved vector with zeros:\n", Z_interleaved)

# ==========================================
# Day 8: NumPy Questions 71 - 80
# ==========================================
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# 71. Multiply an array of dimension (5,5,3) by an array with dimensions (5,5)
image = np.ones((5, 5, 3))
mask = np.random.randint(0, 2, (5, 5))
mask_3d = mask[..., np.newaxis]  # Expand dimensions for broadcasting
result = image * mask_3d
print("71. Multiplied (5,5,3) by (5,5) mask. Result shape:", result.shape)

# 72. How to swap two rows of an array?
A_swap = np.arange(25).reshape(5, 5)
A_swap[[0, 1]] = A_swap[[1, 0]]
print("\n72. Array with Row 0 and Row 1 swapped:\n", A_swap[:2])  # Printing just top 2 rows for brevity

# 73. Find the set of unique line segments composing all triangles
faces = np.random.randint(0, 100, (10, 3))
F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
F = F.reshape(len(F)*3, 2)
F = np.sort(F, axis=1)
G = F.view(dtype=[('p0', F.dtype), ('p1', F.dtype)])
G_unique = np.unique(G)
print("\n73. Unique line segments from triangles:\n", G_unique[:5])  # Showing first 5

# 74. Given a bincount array C, produce array A such that np.bincount(A) == C
C_bin = np.array([1, 1, 2, 3, 4, 4, 6])
A_orig = np.repeat(np.arange(len(C_bin)), C_bin)
print("\n74. Reconstructed Array (A):", A_orig)

# 75. How to compute averages using a sliding window over an array?
Z_window = np.array([1, 2, 3, 4, 5, 6, 7])
windows_avg = sliding_window_view(Z_window, window_shape=3).mean(axis=-1)
print("\n75. Sliding window averages:", windows_avg)

# 76. Build a 2D array with shifted rows using sliding window
windows_2d = sliding_window_view(Z_window, window_shape=3)
print("\n76. 2D array from sliding window:\n", windows_2d)

# 77. How to negate a boolean, or to change the sign of a float inplace?
Z_bool = np.random.randint(0, 2, 5)  # Kept small
np.logical_not(Z_bool, out=Z_bool)
Z_float = np.random.uniform(-1.0, 1.0, 5)
np.negative(Z_float, out=Z_float)
print("\n77. Inplace negations successful.")

# 78. Compute distance from a point p to each line (P0, P1)
P0 = np.random.uniform(-10, 10, (5, 2))
P1 = np.random.uniform(-10, 10, (5, 2))
p = np.array([0, 0])
distances_to_p = np.abs(np.cross(P1-P0, p-P0)) / np.linalg.norm(P1-P0, axis=1)
print("\n78. Distances from origin to lines:", np.round(distances_to_p, 2))

# 79. Compute distance from each point in P to each line (P0, P1)
P_set = np.random.uniform(-10, 10, (10, 2))
vec_point = P_set[:, np.newaxis, :] - P0
vec_line = P1 - P0
area = np.abs(vec_line[:, 0] * vec_point[:, :, 1] - vec_line[:, 1] * vec_point[:, :, 0])
base_length = np.linalg.norm(vec_line, axis=1)
distances_matrix = area / base_length
print("\n79. Distance matrix (Points to Lines) shape:", distances_matrix.shape)

# 80. Extract a subpart with a fixed shape and centered on a given element (padded)


def extract_subpart(image, center, frame_shape, fill_value=0):
    canvas = np.full(frame_shape, fill_value)
    cx, cy = center
    fh, fw = frame_shape
    rx, ry = fh // 2, fw // 2

    src_x_start = max(0, cx - rx)
    src_x_stop = min(image.shape[0], cx - rx + fh)
    src_y_start = max(0, cy - ry)
    src_y_stop = min(image.shape[1], cy - ry + fw)

    dst_x_start = rx - (cx - src_x_start)
    dst_x_stop = dst_x_start + (src_x_stop - src_x_start)
    dst_y_start = ry - (cy - src_y_start)
    dst_y_stop = dst_y_start + (src_y_stop - src_y_start)

    canvas[dst_x_start:dst_x_stop, dst_y_start:dst_y_stop] = image[src_x_start:src_x_stop, src_y_start:src_y_stop]
    return canvas


Z_grid = np.arange(1, 26).reshape(5, 5)
frame = extract_subpart(Z_grid, center=(0, 0), frame_shape=(3, 3), fill_value=0)
print("\n80. Cropped 3x3 Frame (Center at top-left '1', padded with 0):\n", frame)

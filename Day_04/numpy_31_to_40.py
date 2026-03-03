# ==========================================
# Day 4: NumPy Questions 31 - 40
# ==========================================
import numpy as np

# 31. How to ignore all numpy warnings (not recommended)?
# Note: It is safer to only ignore specific warnings, like 'divide' or 'invalid'
with np.errstate(divide='ignore'):
    result = np.array([1]) / 0
print("31. Ignored divide by zero warning. Result:", result)

# 32. Is the following expression true?
print("\n32. np.sqrt(-1) == np.emath.sqrt(-1):", np.sqrt(-1) == np.emath.sqrt(-1))
# False because emath automatically switches the domain to complex numbers.

# 33. How to get the dates of yesterday, today and tomorrow?
today = np.datetime64('today', 'D')
yesterday = today - np.timedelta64(1, 'D')
tomorrow = today + np.timedelta64(1, 'D')
print(f"\n33. Yesterday: {yesterday}, Today: {today}, Tomorrow: {tomorrow}")

# 34. How to get all the dates corresponding to the month of July 2016?
july_2016 = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print("\n34. July 2016 Dates:\n", july_2016)

# 35. How to compute ((A+B)*(-A/2)) in place (without copy)?
A = np.array([1.0, 2.0, 3.0])
B = np.array([4.0, 5.0, 6.0])
np.add(A, B, out=B)           # B becomes (A+B)
np.divide(A, 2, out=A)        # A becomes (A/2)
np.negative(A, out=A)         # A becomes (-A/2) -> Fixed the in-place bug here!
np.multiply(A, B, out=A)      # A becomes (-A/2) * (A+B)
print("\n35. In-place computation result:", A)

# 36. Extract the integer part of a random array of positive numbers using 4 methods
Z = np.random.uniform(0, 10, 5)
print("\n36. Original Array:  ", Z)
print("    Method 1 (Mod):  ", Z - Z % 1)
print("    Method 2 (Floor):", np.floor(Z))
print("    Method 3 (Trunc):", np.trunc(Z))
print("    Method 4 (Type): ", Z.astype(int))

# 37. Create a 5x5 matrix with row values ranging from 0 to 4
arr_zeros = np.zeros((5, 5))
row_vals = np.arange(5)
matrix_5x5 = arr_zeros + row_vals  # Utilizing broadcasting
print("\n37. 5x5 Matrix via Broadcasting:\n", matrix_5x5)

# 38. Consider a generator function that generates 10 integers and use it to build an array


def generate_integers():
    for x in range(10):
        yield x


Z_iter = np.fromiter(generate_integers(), dtype=int, count=-1)
print("\n38. Array from generator:", Z_iter)

# 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded
arr_linspace = np.linspace(0, 1, 12)[1:-1]
print("\n39. Vector (0 to 1 excluded):\n", arr_linspace)

# 40. Create a random vector of size 10 and sort it
arr_rand = np.random.randint(-10, 20, 10)
print("\n40. Original Random Vector:", arr_rand)
print("    Sorted Vector:         ", np.sort(arr_rand))

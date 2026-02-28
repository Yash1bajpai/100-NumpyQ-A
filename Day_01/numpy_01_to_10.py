# Q1. Import the numpy package under the name `np`
import numpy as np

arr = np.array(1)
print(arr)

# Q2. Print the numpy version and the configuration
print(np.__version__, np.show_config())

# Q3. Create a null vector of size 10
arr1 = np.zeros(10)
print(arr1)

# Q4. How to find the memory size of any array
arr2 = np.array([1, 2, 3, 4, 5])
print(arr2.nbytes)
print(arr2.itemsize)

# 5. How to get the documentation of the numpy add function from the command line? (★☆☆)
# Ans. ```python
# %run `python -c "import numpy; numpy.info(numpy.add)"`
# ```

# Q6. Create a null vector of size 10 but the fifth value which is 1
arr3 = np.zeros(10)
arr3[4] = 1
print(arr3)

# Q7. Create a vector with values ranging from 10 to 49
arr4 = np.arange(10, 50)
print(arr4)

# Q8. Reverse a vector (first element becomes last)
print(arr4[-1::-1])

# Q9. Create a 3x3 matrix with values ranging from 0 to 8
arr5 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(arr5)
arr6 = np.arange(0, 9).reshape(3, 3)
print(arr6)

# Q10. Find indices of non-zero elements from [1,2,0,0,4,0]
arr7 = np.array([1, 2, 0, 0, 4, 0])
print(arr7)
index = [i for i in range(len(arr7)) if arr7[i] != 0]  # just use loop for the comparison
print(index)
print(np.nonzero([1, 2, 0, 0, 4, 0]))

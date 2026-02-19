import numpy as np
# Exercise 1:
# Assuming I'm allowed to use numpy
arr = np.arange(1, 100)
standard_deviation = np.std(arr)
average = np.average(arr)

# Exercise 2:
# I assumed in the previous question that 'until 100' meant not including 100
even_array = arr[np.where(arr%2==0)] # np.where returns indices, and since we start with value 1 at index 0, we have to use these indices to get the array values, instead of using the indices directly
odd_array = arr[np.where(arr%2!=0)]

even_standard_deviation = np.std(even_array)
even_average = np.average(even_array)

odd_standard_deviation = np.std(odd_array)
odd_average = np.average(odd_array)

# Exercise 3: exclude [10.20] and [45,57] from average and std
exclusion_range = np.concatenate((np.arange(10,20), np.arange(45,57)))
arr_filtered = arr[~np.isin(arr, exclusion_range)]

excluded_standard_deviation = np.std(arr_filtered) 
excluded_average = np.average(arr_filtered)

# Exercise 4:
import matplotlib.pyplot as plt
x = np.linspace(0,3,num=100)
y = 0.8*np.exp(x) - 2*x

plt.figure()
plt.plot(x,y)
plt.show()

# Exercise 5:
from math import factorial
k_range = range(5)
x = np.linspace(0,3, num=10)
e_approx = np.sum(x**k/factorial(k) for k in k_range)

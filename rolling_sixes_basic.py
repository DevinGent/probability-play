import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def count_sixes(test_size):
    """Takes an integer and rolls a dice test_size times, 
    before returning the number of 6s rolled."""
    number_sixes=0
    for i in range(test_size):
        if np.random.randint(1,7)==6:
            number_sixes=number_sixes+1
    return number_sixes/test_size


# Testing how to apply a function to each element of an array.
# Create a NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Define a  function to square elements
def square(x):
    return x**2

# Pass the NumPy array into the function
squared_arr = square(arr)

print(squared_arr)

rolls = np.array([x for x in range(10,10000,10)])
sixes_percent = np.vectorize(count_sixes)(rolls)
print(sixes_percent)
sns.lineplot(x=rolls,y=sixes_percent)
plt.show()
# As the number of rolls go up, the percent of sixes rolled comes closer to around 16%.
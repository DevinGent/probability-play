import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import math
import matplotlib.ticker as mtick

# We will manually create a binomial distribution (corresponding to the number of sixes rolled when rolling a fair die n times)
# and compare the result to the binomial distribution provided by numpy.


def count_sixes(test_size):
    """Takes an integer and rolls a dice test_size times, 
    before returning the number of 6s rolled."""
    number_sixes=0
    for i in range(test_size):
        if np.random.randint(1,7)==6:
            number_sixes=number_sixes+1
    return number_sixes
#############################################################
np.random.seed(1)
# We set the seed to make the results reproducible.

# Testing rolling dice.
successes=0
for i in range(10):
    roll=np.random.randint(1,7)
    print(roll)
    if roll==6:
        successes=successes+1
    
print("There were {} sixes rolled in 10 attempts.".format(successes))
#################################################################################



TEST_SIZE = 1000
# How many times the dice are rolled in each test.  

NUMBER_OF_TESTS =100 
# This gives how many large the distribution is, i.e., how many times the test of rolling a die n times is performed.

print("In each test a die will be rolled {} times, and the number of sixes rolled will be recorded.".format(TEST_SIZE))
print("The test will be repeated {} times.".format(NUMBER_OF_TESTS))


# We will denote our distribution by an upper case X.
X = np.array([count_sixes(TEST_SIZE) for i in range(NUMBER_OF_TESTS)])
print(X)

# We will see how this compares to a binomial and normal distribution.  First let's graph.
sns.set_style("darkgrid")
sns.kdeplot(X)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.xlabel("Number of sixes rolled in test")
plt.title("X")
plt.show()


binomial=np.random.binomial(TEST_SIZE,1/6,NUMBER_OF_TESTS)
normal = np.random.normal(loc=TEST_SIZE/6,scale=X.std(),size=NUMBER_OF_TESTS)

print("The sizes of the three distributions are {}, {}, and {}".format(len(X),len(binomial),len(normal)))


# Testing that drawing numbers from a binomial distribution works.
print(np.random.binomial(TEST_SIZE,1/6,1))
print(np.random.binomial(TEST_SIZE,1/6,1))
print(np.random.binomial(TEST_SIZE,1/6,1))
print(np.random.binomial(TEST_SIZE,1/6,1))
print(np.random.binomial(TEST_SIZE,1/6,1))



plt.figure(figsize=(8,6))
sns.kdeplot(X, label='X')
sns.kdeplot(binomial, label="Binomial", alpha=.7)
sns.kdeplot(normal,label='Normal',alpha=.7)
plt.text(.03, .85, ' p$\\approx.16$ \n n$={}$ \n size$={}$'.format(TEST_SIZE, NUMBER_OF_TESTS),
        bbox={'boxstyle':'round','fc': 'lightgrey', 'ec':'darkgrey', 'alpha': 0.5, 'pad': .3},ha='left',transform=plt.gca().transAxes)
plt.title("Comparing Distributions")
plt.legend()
plt.show()



# In a normal distribution 68% of values are within 1 standard deviation of the mean,
# 95% of values are within 2 standard deviations of the mean,
# and 99.7% of values are within 3 standard deviations of the mean.
std=X.std()
mean=X.mean()
print("The mean of our distribution is {} and the standard deviation is {}".format(mean, std))




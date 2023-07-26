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

# We are going to make a dataframe with three columns.  The first column will include a test size (how many times a die is rolled),
# the second will show how many 6s were rolled during the test,
# and the third will show what percent of the rolls during that test were 6s.

# Testing rolling dice.
successes=0
for i in range(10):
    roll=np.random.randint(1,7)
    print(roll)
    if roll==6:
        successes=successes+1
    
print("There were {} sixes rolled in 10 attempts.".format(successes))

TEST_SIZE = 100
# How many times the dice are rolled in each experiment.  

NUMBER_OF_TESTS =1000 
# This gives how many large the distribution is, i.e., how many times the experiment of rolling a die n times is performed.

print("In each experiment a die will be rolled {} times, and the number of sixes rolled will be recorded.".format(TEST_SIZE))
print("The experiment will be repeated {} times.".format(NUMBER_OF_TESTS))


# We will denote our distribution by an upper case X.
X = np.array([count_sixes(TEST_SIZE) for i in range(NUMBER_OF_TESTS)])
print(X)

# We will see how this compares to a binomial and normal distribution.  First let's graph.

sns.kdeplot(X, label='X')

plt.legend()
plt.show()

binomial=np.random.binomial(TEST_SIZE,1/6,NUMBER_OF_TESTS)



# Now let us compare this to a binomial distribution.
plt.figure(figsize=(6,4))
sns.kdeplot(X)
print(np.random.binomial(TEST_SIZE,1/6,1))
print(np.random.binomial(TEST_SIZE,1/6,1))
sns.kdeplot(np.random.binomial(TEST_SIZE,1/6,NUMBER_OF_TESTS))
plt.show()

plt.figure(figsize=(8,6))
sns.kdeplot(X, label='X')
sns.kdeplot(binomial, label="Binomial")
plt.text(.03, .85, ' p$\\approx.16$ \n n$={}$ \n size$={}$'.format(TEST_SIZE, NUMBER_OF_TESTS),
        bbox={'boxstyle':'round','facecolor': 'white', 'alpha': 0.5, 'pad': 1},ha='left',transform=plt.gca().transAxes)
plt.legend()
plt.show()



# In a normal distribution 68% of values are within 1 standard deviation of the mean,
# 95% of values are within 2 standard deviations of the mean.
# and 99.7% of values are within 3 standard deviations of the mean
std=X.std()
mean=X.mean()
print("The mean is {} and the standard deviation is {}".format(mean, std))
"""
within_one=df[(df['Percent']>=(mean-std))&(df['Percent']<=(mean+std))]
within_one
print(within_one)
print("The size is {}".format(within_one.shape[0]))
print("{}% of the values are within one standard deviation of the mean.".format(round(100*within_one.shape[0]/df.shape[0],2)))

within_two=df[(df['Percent']>=(mean-2*std))&(df['Percent']<=(mean+2*std))]
within_two.info()
print(within_two)
print("The size is {}".format(within_two.shape[0]))
print("{}% of the values are within two standard deviations of the mean.".format(round(100*within_two.shape[0]/df.shape[0],2)))

within_three=df[(df['Percent']>=(mean-3*std))&(df['Percent']<=(mean+3*std))]
within_three.info()
print(within_three)
print("The size is {}".format(within_three.shape[0]))
print("{}% of the values are within three standard deviations of the mean.".format(round(100*within_three.shape[0]/df.shape[0],2)))

"""



# Now suppose we want to see how increasing the test size effects the percent of sixes rolled.
df = pd.DataFrame({'Test Size':[i+1 for i in range(NUMBER_OF_TESTS)]})
print(df)
df['Sixes']=[count_sixes(i) for i in df['Test Size']]
print(df)
df['Percent']=round(df['Sixes']/df['Test Number'],4)
print(df)
print(df['Percent'].describe())

plt.figure(figsize=(8,6))
sns.lineplot(data=df,x='Test Size',y='Percent')
plt.show(block=False)

# If I want to turn the y axis to percentages.
plt.figure(figsize=(8,6))
sns.lineplot(data=df,x='Test Size',y='Percent')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.show()



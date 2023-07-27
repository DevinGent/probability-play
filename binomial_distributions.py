import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import time

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



TEST_SIZE = 100
# How many times the dice are rolled in each test.  

NUMBER_OF_TESTS =1000 
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


within_one=[i for i in X if ((i>=(mean-std))&(i<=(mean+std)))]
print(within_one)
print("The size is {}".format(len(within_one)))
print("{}% of the values are within one standard deviation of the mean.".format(round(100*len(within_one)/NUMBER_OF_TESTS,2)))

within_two=[i for i in X if ((i>=(mean-2*std))&(i<=(mean+2*std)))]
print(within_two)
print("The size is {}".format(len(within_two)))
print("{}% of the values are within two standard deviations of the mean.".format(round(100*len(within_two)/NUMBER_OF_TESTS,2)))

within_three=[i for i in X if ((i>=(mean-3*std))&(i<=(mean+3*std)))]
print(within_three)
print("The size is {}".format(len(within_three)))
print("{}% of the values are within three standard deviations of the mean.".format(round(100*len(within_three)/NUMBER_OF_TESTS,2)))

#####################################################################################################
# In the construction of X using the code
"""
X = np.array([count_sixes(TEST_SIZE) for i in range(NUMBER_OF_TESTS)])
"""
# we created a distribution which depended on the TEST_SIZE, the NUMBER_OF_TESTS, and an element of randomness. 
# Now we want to see how increasing the TEST_SIZE or NUMBER_OF_TESTS effects the resulting distribution.
# We will define an array of distributions, [X_0, X_1, X_2, X_3, X_4...,X_n], where each distribution has 
# the same TEST_SIZE=100 but an increasing NUMBER_OF_TESTS. 
print("We now consider the case where the test size is fixed, but the number of tests varies.")
# The following code snippet takes an extended time to execute.
print("Please wait as the code executes.  It may take a while.")
start=time.time()
distributions=[]
for ntests in range(50,5001,50):
    distributions.append(np.array([count_sixes(test_size=100) for i in range(ntests)]))
stop=time.time()
print("Generating the distributions took {} seconds".format(stop-start))
# We test that the elements are distributions.
print(distributions[1])

# How do different distribution sizes compare?
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
sns.kdeplot(distributions[0])
plt.gca().set_title("50 tests")
plt.subplot(2,2,2)
sns.kdeplot(distributions[1])
plt.gca().set_title("100 tests")
plt.subplot(2,2,3)
sns.kdeplot(distributions[19])
plt.gca().set_title("1000 tests")
plt.subplot(2,2,4)
sns.kdeplot(distributions[99])
plt.gca().set_title("5000 tests")
plt.show()

# Let's compare this with the binomial and normal distributuon.

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
l1=sns.kdeplot(distributions[0], label='Constructed')
l2=sns.kdeplot(np.random.binomial(100,1/6,50), label='Binomial')
l3=sns.kdeplot(np.random.normal(loc=100/6,scale=distributions[0].std(),size=50),label='Normal')
plt.gca().set_title("50 tests")
plt.subplot(2,2,2)
sns.kdeplot(distributions[1])
sns.kdeplot(np.random.binomial(100,1/6,100))
sns.kdeplot(np.random.normal(loc=100/6,scale=distributions[1].std(),size=100))
plt.gca().set_title("100 tests")
plt.subplot(2,2,3)
sns.kdeplot(distributions[19])
sns.kdeplot(np.random.binomial(100,1/6,1000))
sns.kdeplot(np.random.normal(loc=100/6,scale=distributions[19].std(),size=1000))
plt.gca().set_title("1000 tests")
plt.subplot(2,2,4)
sns.kdeplot(distributions[99])
sns.kdeplot(np.random.binomial(100,1/6,5000))
sns.kdeplot(np.random.normal(loc=100/6,scale=distributions[99].std(),size=5000))
plt.gca().set_title("5000 tests")
plt.gcf().legend([l1,l2,l3],labels=['Constructed', 'Binomial', 'Normal'])
plt.show()







# Let's examine how the mean, standard deviation, median, and interquartile range are affected. 
plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
sns.lineplot(x=[len(dis) for dis in distributions],y=[dis.mean() for dis in distributions])
plt.xlabel('Distribution Size')
plt.ylabel('Mean')
plt.gca().set_title('Mean')

plt.subplot(2,2,2)
sns.lineplot(x=[len(dis) for dis in distributions],y=[dis.std() for dis in distributions])
plt.xlabel('Distribution Size')
plt.ylabel('Standard Deviation')
plt.gca().set_title('Standard Deviation')

plt.subplot(2,2,3)
sns.lineplot(x=[len(dis) for dis in distributions],y=[np.median(dis) for dis in distributions])
plt.xlabel('Distribution Size')
plt.ylabel('Median')
plt.gca().set_title('Median')

plt.subplot(2,2,4)
sns.lineplot(x=[len(dis) for dis in distributions],y=[np.quantile(dis,.75)-np.quantile(dis,.25) for dis in distributions])
plt.xlabel('Distribution Size')
plt.ylabel('IQR')
plt.gca().set_title('Interquartile Range')
plt.tight_layout()
plt.show()

##############################################################################
# Above we considered the case where the number of rolls per test (the TEST_SIZE) was fixed at 100
# and the number of tests increased.  Now let us consider the case where the test size varies, but the
# total number of tests is NUMBER_OF_TESTS=1000.  That is, each distribution is of size 1000.
# We make one change to our approach so far.  Instead of counting how MANY 6s there were over N rolls,
# we will now return what PERCENT of the N rolls came out as 6.
print("We now consider the case where the number of rolls per test varies, but the total number of tests is always 1000.")
print("Please wait as the code executes.  It may take a while.")
start=time.time()
distributions=[]
for test_size in range(10,1001,10):
    distributions.append(np.array([count_sixes(test_size)/test_size for i in range(1000)]))
stop=time.time()
print("Generating the distributions took {} seconds".format(stop-start))
# How do different test sizes compare?
plt.figure(figsize=(10,8))
ax1=plt.subplot(2,2,1)
sns.kdeplot(distributions[0])
ax1.set_title("10 rolls per test")
ax1.set_xlim(0,1)
ax1.set_ylim(0,3)
plt.subplot(2,2,2)
sns.kdeplot(distributions[9])
plt.gca().set_title("100 rolls per test")
plt.subplot(2,2,3)
sns.kdeplot(distributions[49])
plt.gca().set_title("500 rolls per test")
plt.subplot(2,2,4)
sns.kdeplot(distributions[99])
plt.gca().set_title("1000 rolls per test")
plt.show()

"""
plt.figure(figsize=(8,6))
sns.lineplot(data=df,x='Test Size',y='Percent')
plt.show(block=False)

# If I want to turn the y axis to percentages.
plt.figure(figsize=(8,6))
sns.lineplot(data=df,x='Test Size',y='Percent')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.show()
"""


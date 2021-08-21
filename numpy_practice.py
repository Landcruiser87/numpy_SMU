# https://www.machinelearningplus.com/python/101-numpy-exercises-python/



# 1. Import numpy as np and see the version
# Difficulty Level: L1
# Q. Import numpy as np and print the version number.

##? 1. Import numpy as np and see the version
# Difficulty Level: L1

# Q. Import numpy as np and print the version number.

import numpy as np
print(np.__version__)

##? 2. How to create a 1D array?
# Difficulty Level: L1

# Q. Create a 1D array of numbers from 0 to 9

arr = np.arange(10)
arr

##? 3. How to create a boolean array?
# Difficulty Level: L1

# Q. Create a 3×3 numpy array of all True’s

arr = np.full((3,3), True, dtype=bool)
arr

##? 4. How to extract items that satisfy a given condition from 1D array?
# Difficulty Level: L1

# Q. Extract all odd numbers from arr

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

arr[arr % 2 == 1]

##? 5. How to replace items that satisfy a condition with another value in numpy array?
# Difficulty Level: L1

# Q. Replace all odd numbers in arr with -1
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

arr[arr % 2 == 1] = -1
arr

##? 6. How to replace items that satisfy a condition without affecting the original array?
# Difficulty Level: L2

# Q. Replace all odd numbers in arr with -1 without changing arr

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

out = np.where(arr % 2 == 1, -1, arr)

out
arr

##? 7. How to reshape an array?
# Difficulty Level: L1

# Q. Convert a 1D array to a 2D array with 2 rows

arr = np.arange(10)

arr.reshape(2, -1)
# Setting y to -1 automatically decides number of columns.  
# Could do the same with 
arr.reshape(2, 5)

##? 8. How to stack two arrays vertically?
# Difficulty Level: L2

# Q. Stack arrays a and b vertically
a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)

#1
np.vstack([a, b])
#2
np.concatenate([a, b], axis=0)
#3
np.r_[a, b]

# 9. How to stack two arrays horizontally?
# Difficulty Level: L2

# Q. Stack the arrays a and b horizontally.

a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)

#1
np.hstack([a, b])
#2
np.concatenate([a, b], axis=1)
#3
np.c_[a, b]

##? 10. How to generate custom sequences in numpy without hardcoding?
# Difficulty Level: L2

# Q. Create the following pattern without hardcoding. 
# Use only numpy functions and the below input array a.

a = np.array([1,2,3])

np.r_[np.repeat(a,3), np.tile(a, 3)]

##? 11. How to get the common items between two python numpy arrays?
# Difficulty Level: L2

# Q. Get the common items between a and b
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

np.intersect1d(a, b)

##? 12. How to remove from one array those items that exist in another?
# Difficulty Level: L2

# Q. From array a remove all items present in array b

a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])

# From 'a' remove all of 'b'
np.setdiff1d(a,b)

##? 13. How to get the positions where elements of two arrays match?
# Difficulty Level: L2

# Q. Get the positions where elements of a and b match

a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

np.where(a==b)

# 14. How to extract all numbers between a given range from a numpy array?
# Difficulty Level: L2

# Q. Get all items between 5 and 10 from a.

a = np.array([2, 6, 1, 9, 10, 3, 27])

#1 
idx = np.where((a>=5) & (a<=10))
a[idx]

#2
idx = np.where(np.logical_and(a >= 5, a <= 10))
a[idx]

#3
a[(a >= 5) & (a <= 10)]


##? 15. How to make a python function that handles scalars to work on numpy arrays?
# Difficulty Level: L2

# Q. Convert the function maxx that works on two scalars, to work on two arrays.

def maxx(x:np.array, y:np.array):
	"""Get the maximum of two items"""
	if x >= y:
		return x
	else:
		return y

a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
pair_max = np.vectorize(maxx, otypes=[float])

pair_max(a, b)

##? 16. How to swap two columns in a 2d numpy array?
# Difficulty Level: L2

# Q. Swap columns 1 and 2 in the array arr.

arr = np.arange(9).reshape(3,3)
arr

arr[:, [1, 0, 2]]
#by putting brackets inside the column slice.  You have access to column indices



##? 17. How to swap two rows in a 2d numpy array?
# Difficulty Level: L2

# Q. Swap rows 1 and 2 in the array arr:

arr = np.arange(9).reshape(3,3)
arr

arr[[0, 2, 1], :]
#same goes here for the rows

##? 18. How to reverse the rows of a 2D array?
# Difficulty Level: L2

# Q. Reverse the rows of a 2D array arr.

# Input
arr = np.arange(9).reshape(3,3)
arr

arr[::-1, :]

#or
arr[::-1]

# 19. How to reverse the columns of a 2D array?
# Difficulty Level: L2

# Q. Reverse the columns of a 2D array arr.

# Input
arr = np.arange(9).reshape(3,3)
arr

arr[:,::-1]

##? 20. How to create a 2D array containing random floats between 5 and 10?
# Difficulty Level: L2

# Q. Create a 2D array of shape 5x3 to contain random decimal numbers between 5 and 10.
arr = np.arange(9).reshape(3,3)

#1
rand_arr = np.random.randint(low=5, high=10, size=(5,3)) + np.random.random((5,3))
rand_arr

#2
rand_arr = np.random.uniform(5, 10, size=(5,3))
rand_arr
##? 21. How to print only 3 decimal places in python numpy array?
# Difficulty Level: L1

# Q. Print or show only 3 decimal places of the numpy array rand_arr.

rand_arr = np.random.random((5,3))
rand_arr

rand_arr = np.random.random([5,3])
np.set_printoptions(precision=3)
rand_arr[:4]

##? 22. How to pretty print a numpy array by suppressing the scientific notation (like 1e10)?
# Difficulty Level: L1

# Q. Pretty print rand_arr by suppressing the scientific notation (like 1e10)

#Reset printoptions
np.set_printoptions(suppress=False)

# Create the random array
np.random.seed(100)
rand_arr = np.random.random([3,3])/1e3
rand_arr

#Set precision and suppress e notation
np.set_printoptions(suppress=True, precision=6)
rand_arr

##? 23. How to limit the number of items printed in output of numpy array?
# Difficulty Level: L1

# Q. Limit the number of items printed in python numpy array a to a maximum of 6 elements.

a = np.arange(15)

#set the elements to print in threshold
np.set_printoptions(threshold=6)
a

# reset the threshold to default
np.set_printoptions(threshold=1000)

##? 24. How to print the full numpy array without truncating
# Difficulty Level: L1

# Q. Print the full numpy array a without truncating.

a = np.arange(15)
# reset the threshold to default
np.set_printoptions(threshold=1000)
a


##? 25. How to import a dataset with numbers and texts keeping the text intact in python numpy?
# Difficulty Level: L2

# Q. Import the iris dataset keeping the text intact.

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype="object")
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

iris[:3]

##? 26. How to extract a particular column from 1D array of tuples?
# Difficulty Level: L2

# Q. Extract the text column species from the 1D iris imported in previous question.

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None, encoding = "UTF-8")

species = np.array([col[4] for col in iris_1d])
species[:5]

##? 27. How to convert a 1d array of tuples to a 2d numpy array?
# Difficulty Level: L2

# Q. Convert the 1D iris to 2D array iris_2d by omitting the species text field.

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None, encoding = "UTF-8")

#1
no_species_2d = np.array([row.tolist()[:4] for row in iris_1d])
no_species_2d[:3]

#2
# Can directly specify columns to use with the "usecols" method
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
no_species_2d = np.genfromtxt(url, delimiter=',', dtype=None, encoding = "UTF-8", usecols=[0,1,2,3])
no_species_2d[:3]

##? 28. How to compute the mean, median, standard deviation of a numpy array?
# Difficulty: L1

# Q. Find the mean, median, standard deviation of iris's sepallength (1st column)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None, encoding="utf-8")

sepal = np.genfromtxt(url, delimiter=',', dtype=float, usecols=[0])
# or
sepal = np.array([col[0] for col in iris_1d])
# or
sepal = np.array([col.tolist()[0] for col in iris_1d])

mu, med, sd = np.mean(sepal), np.median(sepal), np.std(sepal)
np.set_printoptions(precision=2)
print(f'The mean is {mu} \nThe median is {med} \nThe standard deviation is {sd}')

##? 29. How to normalize an array so the values range exactly between 0 and 1?
# Difficulty: L2

# Q. Create a normalized form of iris's sepallength whose values range exactly between 0 and 1 so that the minimum has value 0 and maximum has value 1.
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None, encoding="utf-8")

sepal = np.genfromtxt(url, delimiter=',', dtype=float, usecols=[0])

#1
smax, smin = np.max(sepal), np.min(sepal)
S = (sepal-smin)/(smax-smin)
S 

#2
S = (sepal-smin)/sepal.ptp()
S

##? 30. How to compute the softmax score?
# Difficulty Level: L3

# Q. Compute the softmax score of sepallength.

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

sepal = np.genfromtxt(url, delimiter=',', dtype=float, usecols=[0], encoding="utf-8")

#or

sepal = np.genfromtxt(url, delimiter=',', dtype='object')
sepal = np.array([float(row[0]) for row in sepal])
# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python"""

#1
def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x/ e_x.sum(axis=0)

softmax(sepal)

##? 31. How to find the percentile scores of a numpy array?
# Difficulty Level: L1

# Q. Find the 5th and 95th percentile of iris's sepallength

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepal = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

np.percentile(sepal, q=[5, 95])

##? 32. How to insert values at random positions in an array?
# Difficulty Level: L2

# Q. Insert np.nan values at 20 random positions in iris_2d dataset

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', encoding="utf-8")
#Can change object to float if you want

#1 
i, j = np.where(iris_2d)
# i, j contain the row numbers and column numbers of the 600 elements of Irix_x
np.random.seed(100)
iris_2d[np.random.choice(i, 20), np.random.choice((j), 20)] =  np.nan

#Checking nans in 2nd column
np.isnan(iris_2d[:, 1]).sum()

#Looking over all rows/columns
np.isnan(iris_2d[:, :]).sum()

#2
np.random.seed(100)
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)]=np.nan

#Looking over all rows/columns
np.isnan(iris_2d[:, :]).sum()

##? 33. How to find the position of missing values in numpy array?
# Difficulty Level: L2

# Q. Find the number and position of missing values in iris_2d's sepallength (1st column)

# ehh already did that?  Lol.  Using above filtered array from method 2 in
# question 32

np.isnan(iris_2d[:, 0]).sum()

#Indexes of which can be found with
np.where(np.isnan(iris_2d[:, 0]))

##? 34. How to filter a numpy array based on two or more conditions?
# Difficulty Level: L3

# Q. Filter the rows of iris_2d that has petallength (3rd column) > 1.5 
# and sepallength (1st column) < 5.0
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

filt_cond = (iris_2d[:,0] < 5.0) & (iris_2d[:, 2] > 1.5)

iris_2d[filt_cond]


##? 35. How to drop rows that contain a missing value from a numpy array?
# Difficulty Level: L3:

# Q. Select the rows of iris_2d that does not have any nan value.

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

#1
#No direct numpy implementation
iris_drop = np.array([~np.any(np.isnan(row)) for row in iris_2d])
#Look at first 5 rows of drop
iris_2d[iris_drop][:5]

#2
iris_2d[np.sum(np.isnan(iris_2d), axis=1)==0][:5]


##? 36. How to find the correlation between two columns of a numpy array?
# Difficulty Level: L2

# Q. Find the correlation between SepalLength(1st column) and PetalLength(3rd column) in iris_2d

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

#1
np.corrcoef(iris_2d[:, 0], iris_2d[:, 2])[0, 1]

#2
from scipy.stats.stats import pearsonr
corr, p_val = pearsonr(iris_2d[:, 0], iris_2d[:, 2])
print(corr)

# Correlation coef indicates the degree of linear relationship between two numeric variables.
# It can range between -1 to +1.

# The p-value roughly indicates the probability of an uncorrelated system producing 
# datasets that have a correlation at least as extreme as the one computed.
# The lower the p-value (<0.01), greater is the significance of the relationship.
# It is not an indicator of the strength.
#> 0.871754157305



##? 37. How to find if a given array has any null values?
# Difficulty Level: L2

# Q. Find out if iris_2d has any missing values.

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

np.isnan(iris_2d[:, :]).any()


##? 38. How to replace all missing values with 0 in a numpy array?
# Difficulty Level: L2

# Q. Replace all occurrences of nan with 0 in numpy array
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

#Check for nans
np.any(~np.isnan(iris_2d[:, :]))
#Set Indexes of of the nans = 0
iris_2d[np.isnan(iris_2d)] = 0

#Check the same indexes
np.where(iris_2d==0)

#Check first 10 rows
iris_2d[:10]

##? 39. How to find the count of unique values in a numpy array?
# Difficulty Level: L2

# Q. Find the unique values and the count of unique values in iris's species

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object', encoding="utf-8")
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

#1
species = np.array([row.tolist()[4] for row in iris])
np.unique(species, return_counts=True)

#2
np.unique(iris[:, 4], return_counts=True)


##? 40. How to convert a numeric to a categorical (text) array?
# Difficulty Level: L2

# Q. Bin the petal length (3rd) column of iris_2d to form a text array, such that if petal length is:

# Less than 3 --> 'small'
# 3-5 --> 'medium'
# '>=5 --> 'large'

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

#1
#Bin the petal length
petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])

#Map it to respective category. 
label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}
petal_length_cat = [label_map[x] for x in petal_length_bin]
petal_length_cat[:4]

#or

petal_length_cat = np.array(list(map(lambda x: label_map[x], petal_length_bin)))
petal_length_cat[:4]


##? 41. How to create a new column from existing columns of a numpy array?
# Difficulty Level: L2

# Q. Create a new column for volume in iris_2d, 
# where volume is (pi x petallength x sepal_length^2)/3

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')

# Compute volume
sepallength = iris_2d[:, 0].astype('float')
petallength = iris_2d[:, 2].astype('float')

volume = (np.pi * petallength*sepallength**2)/3


# Introduce new dimension to match iris_2d's
volume = volume[:, np.newaxis]
# Add the new column
out = np.hstack([iris_2d, volume])

out[:4]


##? 42. How to do probabilistic sampling in numpy?
# Difficulty Level: L3

# Q. Randomly sample iris's species such that setosa 
# is twice the number of versicolor and virginica

# Import iris keeping the text column intact
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

#Get species column
species = iris[:, 4]

#1 Generate Probablistically.
np.random.seed(100)
a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])

#Checking counts
np.unique(out[:], return_counts=True)

#2 Probablistic Sampling #preferred
np.random.seed(100)
probs = np.r_[np.linspace(0, 0.500, num=50), np.linspace(0.501, .0750, num=50), np.linspace(.751, 1.0, num=50)]
index = np.searchsorted(probs, np.random.random(150))
species_out = species[index]
print(np.unique(species_out, return_counts=True))

# Approach 2 is preferred because it creates an index variable that can be 
# used to sample 2d tabular data.


##? 43. How to get the second largest value of an array when grouped by another array?
# Difficulty Level: L2

# Q. What is the value of second longest petallength of species setosa

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')


petal_setosa = iris[iris[:, 4]==b'Iris-setosa', [2]].astype('float')
#1
#Note. Option 1 will return the second largest value 1.7, but with no repeats (np.unique()
np.unique(np.sort(petal_setosa))[-2]

#Note, options 2 and 3. these will return 1.9 because that is the second largest value.
#2
petal_setosa[np.argpartition(petal_setosa, -2)[-2]]


#3
petal_setosa[petal_setosa.argsort()[-2]]

#4
unq = np.unique(petal_setosa)
unq[np.argpartition(unq, -2)[-2]]
#Note:  This method still gives back 1.9.  As that is the 2nd largest value,
#So you'd have to filter for unique values.  Then do the argpart on the unq array



##? 44. How to sort a 2D array by a column
# Difficulty Level: L2

# Q. Sort the iris dataset based on sepallength column.

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# dtype = [('sepallength', float), ('sepalwidth', float), ('petallength', float), ('petalwidth', float),('species', 'S10')]
iris = np.genfromtxt(url, delimiter=',', dtype="object")
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

#1
print(iris[iris[:,0].argsort()][:20])

#2
#!Only captures first column to sort
np.sort(iris[:, 0], axis=0)

#3
sorted(iris, key=lambda x: x[0])

##? 45. How to find the most frequent value in a numpy array?
# Difficulty Level: L1

# Q. Find the most frequent value of petal length (3rd column) in iris dataset.

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

vals, counts = np.unique(iris[:, 2], return_counts=True)
print(vals[np.argmax(counts)])

##? 46. How to find the position of the first occurrence of a value greater than a given value?
# Difficulty Level: L2

# Q. Find the position of the first occurrence of a value greater than 1.0 in petalwidth 4th column of iris dataset.


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

#1
np.argwhere(iris[:, 3].astype(float) > 1.0)[0]


# 47. How to replace all values greater than a given value to a given cutoff?
# Difficulty Level: L2

# Q. From the array a, replace all values greater than 30 to 30 and less than 10 to 10.

np.set_printoptions(precision=2)
np.random.seed(100)
a = np.random.uniform(1,50, 20)

#1
np.clip(a, a_min=10, a_max=30)

#2
np.where(a < 10, 10, np.where(a > 30, 30, a))

#Tangent - Filtering condition
#Say we only want the values above 10 and below 30.  Or operator | should help there.
filt_cond = (a < 10) | (a > 30)
a[filt_cond]

##? 48. How to get the positions of top n values from a numpy array?
# Difficulty Level: L2

# Q. Get the positions of top 5 maximum values in a given array a.

np.random.seed(100)
a = np.random.uniform(1,50, 20)

#1
a.argsort()[:5]

#2
np.argpartition(-a, 5)[:5]

# or (order is reversed though)
np.argpartition(a, -5)[-5:]


#To get the values. 
#1
a[a.argsort()][-5:]

#2 
np.sort(a)[-5:]

#3
np.partition(a, kth=-5)[-5:]

#4

a[np.argpartition(-a, 5)][:5]

#or

a[np.argpartition(a, -5)][-5:]



##? 49. How to compute the row wise counts of all possible values in an array?
# Difficulty Level: L4
# Q. Compute the counts of unique values row-wise.

np.random.seed(100)
arr = np.random.randint(1,11,size=(6, 10))


#1
def row_counts(arr2d):
	count_arr = [np.unique(row, return_counts=True) for row in arr2d]
	return [[int(b[a==i]) if i in a else 0 for i in np.unique(arr2d)] for a, b in count_arr]

print(np.arange(1, 11))

row_counts(arr)

#2
arr = np.array([np.array(list('bill clinton')), np.array(list('narendramodi')), np.array(list('jjayalalitha'))])
print(np.unique(arr))
row_counts(arr)


##? 50. How to convert an array of arrays into a flat 1d array?
# Difficulty Level: 2

# Q. Convert array_of_arrays into a flat linear 1d array.

# Input:
arr1 = np.arange(3)
arr2 = np.arange(3,7)
arr3 = np.arange(7,10)

array_of_arrays = np.array([arr1, arr2, arr3])
array_of_arrays


#1 - List comp
arr_2d = [a for arr in array_of_arrays for a in arr]
arr_2d

#2 - concatenate
arr_2d = np.concatenate([arr1, arr2, arr3])
arr_2d

#3 - hstack
arr_2d = np.hstack([arr1, arr2, arr3])
arr_2d

#4 - ravel
arr_2d = np.concatenate(array_of_arrays).ravel() #ravel flattens the array
arr_2d

##? 51. How to generate one-hot encodings for an array in numpy?
# Difficulty Level L4
# Q. Compute the one-hot encodings (dummy binary variables for each unique value in the array)

# Input

np.random.seed(101) 
arr = np.random.randint(1,4, size=6)
arr

#1
def one_hot_encode(arr):
	uniqs = np.unique(arr)
	out = np.zeros((arr.shape[0], uniqs.shape[0]))
	for i, k in enumerate(arr):
		out[i, k-1] = 1
	return out

print(np.arange(1, 4))
one_hot_encode(arr)

#2
(arr[:, None] == np.unique(arr)).view(np.int8)


##? 52. How to create row numbers grouped by a categorical variable?
# Difficulty Level: L3

# Q. Create row numbers grouped by a categorical variable. 
# Use the following sample from iris species as input.

#Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
#choose 20 species randomly
species_small = np.sort(np.random.choice(species, size=20))
species_small

#1
print([i for val in np.unique(species_small) for i, grp in enumerate(species_small[species_small==val])])


##? 53. How to create group ids based on a given categorical variable?
# Difficulty Level: L4

# Q. Create group ids based on a given categorical variable. 
# Use the following sample from iris species as input.

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
species_small = np.sort(np.random.choice(species, size=20))
species_small

#1
[np.argwhere(np.unique(species_small) == s).tolist()[0][0] for val in np.unique(species_small) for s in species_small[species_small==val]]

#2

# Solution: For Loop version
output = []
uniqs = np.unique(species_small)

for val in uniqs:  # uniq values in group
	for s in species_small[species_small==val]:  # each element in group
		groupid = np.argwhere(uniqs == s).tolist()[0][0]  # groupid
		output.append(groupid)

print(output)


##? 54. How to rank items in an array using numpy?
# Difficulty Level: L2

# Q. Create the ranks for the given numeric array a.

#Input
np.random.seed(10)
a = np.random.randint(20, size=10)
print(a)

a.argsort().argsort()

##? 55. How to rank items in a multidimensional array using numpy?
# Difficulty Level: L3

# Q. Create a rank array of the same shape as a given numeric array a.

#Input
np.random.seed(10)
a = np.random.randint(20, size=[2,5])
print(a)

print(a.ravel().argsort().argsort().reshape(a.shape))

##? 56. How to find the maximum value in each row of a numpy array 2d?
# DifficultyLevel: L2

# Q. Compute the maximum for each row in the given array.

#Input
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
a

#1
[np.max(row) for row in a]

#2
np.amax(a, axis=1)

#3
np.apply_along_axis(np.max, arr=a, axis=1)

##? 57. How to compute the min-by-max for each row for a numpy array 2d?
# DifficultyLevel: L3

# Q. Compute the min-by-max for each row for given 2d numpy array.

#Input
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
a

#1
[np.min(row)/np.max(row) for row in a]

#2
np.apply_along_axis(lambda x: np.min(x)/np.max(x), arr=a, axis=1)


##? 58. How to find the duplicate records in a numpy array?
# Difficulty Level: L3

# Q. Find the duplicate entries (2nd occurrence onwards) in the given numpy array 
# and mark them as True. First time occurrences should be False

# Input
np.random.seed(100)
a = np.random.randint(0, 5, 10)
print('Array: ', a)

#1
def duplicates(arr):
	#Enumerate the array, then compare each element up to that
	#to check for dups
	return [elem in arr[:i] for i, elem in enumerate(arr)]
duplicates(a)

#2
#pythonic version using set (think np.unique() but for sets)
def c_duplicates(X):
	seen = set()
	seen_add = seen.add
	out = []
	for x in X:
		if (x in seen or seen_add(x)):
			out.append(True)
		else:
			out.append(False)
	return out
print(c_duplicates(a))

#3
# Create an all True array
out = np.full(a.shape[0], True)

# Find the index positions of unique elements
unique_positions = np.unique(a, return_index=True)[1]

# Mark those positions as False
out[unique_positions] = False

print(out)


##? 59. How to find the grouped mean in numpy?
# Difficulty Level L3

# Q. Find the mean of a numeric column grouped by a categorical column in 
# a 2D numpy array

#Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

#1
num_col = iris[:, 1].astype('float')
cat_col = iris[:, 4]

[[group_val, num_col[cat_col==group_val].mean()] for group_val in np.unique(cat_col)]

#2 Easier to read
num_col = iris[:, 1].astype('float')
cat_col = iris[:, 4]
all_species = np.unique(cat_col)

[(i, num_col[cat_col==i].mean()) for i in all_species]

#3 For loop
output = []
for group_val in np.unique(cat_col):
	output.append([group_val, num_col[cat_col==group_val].mean()])

output


##? 60. How to convert a PIL image to numpy array?
# Difficulty Level: L3

# Q. Import the image from the following URL and convert it to a numpy array.

#Input
URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'

from io import BytesIO
from PIL import Image
import PIL, requests

# Import image from URL
URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
response = requests.get(URL)

# Read it as Image
I = Image.open(BytesIO(response.content))

# Optionally resize
I = I.resize([150,150])

# Convert to numpy array
arr = np.asarray(I)

# Optionaly Convert it back to an image and show
im = PIL.Image.fromarray(np.uint8(arr))
Image.Image.show(im)

##? 61. How to drop all missing values from a numpy array?
# Difficulty Level: L2

# Q. Drop all nan values from a 1D numpy array

# Input:

a = np.array([1,2,3,np.nan,5,6,7,np.nan])

#1
a[np.logical_not(np.isnan(a))]

#2
a[~np.isnan(a)]

##? 62. How to compute the euclidean distance between two arrays?
# Difficulty Level: L3

# Q. Compute the euclidean distance between two arrays a and b.

# Input:

a = np.array([1,2,3,4,5])
b = np.array([4,5,6,7,8])

#1
dist = np.linalg.norm(a-b)
dist

##? 63. How to find all the local maxima (or peaks) in a 1d array?
# Difficulty Level: L4

# Q. Find all the peaks in a 1D numpy array a. Peaks are points surrounded by smaller values on both sides.

# Input:

a = np.array([1, 3, 7, 1, 2, 6, 0, 1])

#
doublediff = np.diff(np.sign(np.diff(a)))
peak_locations = np.where(doublediff == -2)[0] + 1
peak_locations


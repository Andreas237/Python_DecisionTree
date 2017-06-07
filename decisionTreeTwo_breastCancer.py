from sklearn.tree import DecisionTreeClassifier     # Obvious need
from sklearn import tree                            # export graphviz of the tree
from os import remove as os_rm                      # remove the last created file
import pydotplus                                    # Needed to convert to pdf
from numpy import genfromtxt                        # Convert CSV to array
import time                                         # check how long to build the tree
from decimal import *                               # set decimal precision, and use Decimal


###################################
#       OBJECTIVE
###################################
#
# ADD WRITE UP DONE IN NOTEBOOK
# reference learning problem (1)
# http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.names





test_data_fName='test_wpbc_data2_NoNR.csv'
training_data_fName='training_wpbc_data2_NoNR.csv'





trainingData = genfromtxt( training_data_fName, delimiter=',')
trainingAttrbs = trainingData[:,3:34]
trainingTarget = trainingData[:,35]





testData = genfromtxt( test_data_fName, delimiter=',')
testAttrbs = testData[:,3:34]
testTarget = testData[:,35]



max_depth=3
before = time.clock()
clf3 = DecisionTreeClassifier(random_state=0,
                             max_depth=3)
clf3 = clf3.fit( trainingAttrbs, trainingTarget)
after = time.clock()
print( "Tree build time " + str(max_depth) + ": " + str(after - before) )


max_depth=7
before = time.clock()
clf7 = DecisionTreeClassifier(random_state=0,
                             max_depth=7)
clf7 = clf7.fit( trainingAttrbs, trainingTarget)
after = time.clock()
print( "Tree build time " + str(max_depth) + ": " + str(after - before) )


max_depth=17
before = time.clock()
clf17 = DecisionTreeClassifier(random_state=0,
                             max_depth=17)
clf3 = clf17.fit( trainingAttrbs, trainingTarget)
after = time.clock()
print( "Tree build time " + str(max_depth) + ": " + str(after - before) )



max_depth=56
before = time.clock()
clf56 = DecisionTreeClassifier(random_state=0,
                             max_depth=56)
clf56 = clf17.fit( trainingAttrbs, trainingTarget)
after = time.clock()
print( "Tree build time " + str(max_depth) + ": " + str(after - before) )




clf3TestOutcome = [ clf3.predict(testAttrbs),
                    clf7.predict(testAttrbs),
                    clf17.predict(testAttrbs),
                    clf56.predict(testAttrbs)
                    ]




# Loop throught each of the 4 outcome arrays.
# Check their values against the target outcome array
# Append the precision to correctScore[]

correctScore=[]         # store the scores

getcontext().prec = 15  # set the decimal prevision to 15

for i in range(0,4):
    correct=0
    total=99
    # go through year outcome
    for j in range(0,99):
        correct = correct + ( clf3TestOutcome[i][j] == testTarget[j] )
    correctScore.append( correct )


        
print( correctScore )


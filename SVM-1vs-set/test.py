from svmutil1vset import *

y, x = [1,-1], [{1:1, 3:1}, {1:-1,3:-1}]
prob  = svm_problem(y, x)
param = svm_parameter('-c 4 -b 1')
m = svm_train(prob, param)
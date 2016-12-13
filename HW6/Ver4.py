import numpy as np
import cv2
import matplotlib.pyplot as plt


def predict_self(inp,wji,wkj):

    l0 = inp
    l1 = sigma(np.dot(l0, wji))
    l2 = sigma(np.dot(l1, wkj))
    l2 = abs(l2)
    out = l2;
    for i in range(len(inp)):
        if l2[i]>0.5:
            out[i]=1
        elif l2[i]<0.5:
            out[i]=0
    return out



def sigma(x, deriv=False):
    return 1 / (1 + np.exp(-x))

def sigma_deriv(x):
    return x * (1 - x)

# X = np.array([[0, 0, 1],
#               [0, 1, 1],
#               [1, 0, 1],
#               [1, 1, 1]])
# y = np.array([[0],
#               [1],
#               [1],
#               [0]])

X = np.array([[1,0,1],
             [0,1,1],
             [-1,0,1],
             [0,-1,1],
             [0.5,0.5,1],
             [-0.5,0.5,1],
             [0.5,-0.5,1],
             [-0.5,-0.5,1]
])

y = np.array([
    [1],
    [1],
    [1],
    [1],
    [0],
    [0],
    [0],
    [0],
])

num_inputs = 3 #2 inps includgin the bias
num_HLNodes = 5 #number of nodes in the hidden layer3
num_out = 1 #number of output nodes
epsilon = 0.9 #learning rate
Error = [] #a list to store the errors so that we can plot it
np.random.seed(1)

# for z in range(1,10,1):
z = 0.9
# randomly initialize our weights with mean 0
Wji = 2 * np.random.random((num_inputs, num_HLNodes)) - 1
Wkj = 2 * np.random.random((num_HLNodes, num_out)) - 1
for j in range(80000):
    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = sigma(np.dot(l0, Wji))
    l2 = sigma(np.dot(l1, Wkj))
    # the error of the output
    l2_error = y - l2
    Error.append(np.mean(np.abs(l2_error)))
    # if (j % 10000) == 0:
    #     print ("Error:" + str(np.mean(np.abs(l2_error))))

    # # to print the Error Plot
    # if (j == 79999):
    #     Error.append(np.mean(np.abs(l2_error)))

    # Backpropogation
    l2_delta = l2_error * sigma_deriv(l2)
    del_Wkj = l1.T.dot(l2_delta)
    l1_error = l2_delta.dot(Wkj.T)
    l1_delta = l1_error * sigma_deriv(l1)
    del_Wji = l0.T.dot(l1_delta)

    # updating the weights with respect to the learning rate
    Wkj += del_Wkj * z
    Wji += del_Wji * z



print ("After training")



# create a mesh to plot in
h = .1 #the step size in the mesh plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
inp = np.c_[xx.ravel(), yy.ravel(),np.ones(len(xx.ravel()))]
Z = predict_self(inp,Wji,Wkj)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

# to plot the J vs Epocs
# plt.plot(Error)
# plt.xlabel('Iterations or Epochs(for complete dataset')
# plt.ylabel('J')
# plt.title('The cost vs iterations (or) Learning Curve')
# plt.show()

#Error vs Hidden Layer PE
# yaxis = [i for i in range(3,20,1)]
# plt.plot(Error,yaxis)
# plt.xlabel('Number of hidden layer processing elements')
# plt.ylabel('Final Error')
# plt.title('Hidden Layer Processing Elements vs Error')
# plt.show()

# #Error vs Epsilon plot
# plt.plot(Error)
# plt.xlabel('Epsilon, learning Rate')
# plt.ylabel('Final Error')
# plt.title('Epsilon vs  Learning Error')
# plt.show()

#Input Data plot
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('Input Training Data')
# plt.show()

# # Error vs Epsilon plot
# plt.plot(Error)
# plt.xlabel('Iterations')
# plt.ylabel('Final Error')
# plt.title('Learning Curve')
# plt.show()
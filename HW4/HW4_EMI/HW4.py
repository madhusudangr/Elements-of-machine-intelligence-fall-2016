import numpy


#load data first
loaded_data = numpy.load('data_set.npy')
length = loaded_data.shape[0]
training_data = loaded_data[0:int(0.7 * length),:]
testing_data = loaded_data[int(0.7*length)+1:length,:]

no_of_Classes = 2 # 0 and 1
#the features are seperated based on the clases
class_0 = [training_data[i,1:] for i in range(training_data[:,0].shape[0]) if training_data[i,0] == 0]
class_1 = [training_data[i,1:] for i in range(training_data[:,0].shape[0]) if training_data[i,0] == 1]
#calculate mean
class_mean = numpy.ndarray(shape=(2,7))
class_mean[0,:] = numpy.mean(class_0,axis =0)
class_mean[1,:] = numpy.mean(class_1,axis =0)

# calculating prior
prior = numpy.array((2,1))
prior[0] = len(class_0) / training_data.shape[0]
prior[1] = len(class_1) / training_data.shape[0]

#the classification function
# X = seven input features - the test data
# class_mean - this is a vector which is going to have mean for all features for that particular class
def classify(X):

	#variable to store class prediction
	Predicted_Probablity = numpy.array((2,1))
	for j in range(no_of_Classes):
		
		cov = (X-class_mean[j,:]).transpose() * (X-class_mean[j,:]) #calculate covariance matrix
		P1 = -(1/2)*(X - class_mean[j]).transpose() * numpy.linalg.pinv(cov) * (X - class_mean[j])
		P2 = -(d/2)*log(2*180)
		P3 = -(1/2)*log(det(cov))
		P4 = log(prior[j])
		Predicted_Probablity[j] = P1 + P2 + P3 + P4

	if(Predicted_Probablity[0]>Predicted_Probablity[1]):
		return 0
	else:
		return 1


if __name__ == "__main__":
	for i in range(testing_data.shape[0]):
		X = testing_data[i,:]
		predicted_class[i] = classify(X[1:])




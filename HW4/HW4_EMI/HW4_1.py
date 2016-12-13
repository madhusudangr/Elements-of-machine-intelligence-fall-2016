
import random
import pandas as pd
import numpy

def load_data():
	data = pd.read_csv('HW4.txt', sep = "\t", header = 0)
	data.columns=["Species","FrontalLip","RearWidth","Length","Width","Depth","Male","Female"]
	return data

def prepare_data(data):
	classes = data.Species
	# for c in classes:
	# 	if c == 1:
	# 		num_of_class1 = num_of_class1+1
	# 	else:
	# 		num_of_class0 = num_of_class0+1
	class_0 = [i for i in range(len(data.Species)) if data.Species[i] == 0]
	class_1 = [i for i in range(len(data.Species)) if data.Species[i] == 1]
	num_of_class0 = len(class_0)
	num_of_class1 = len(class_1)
	print("number of crabs ins species 0 :{0} \nnumber of crabs in species 1:{1}").format(num_of_class0,num_of_class1)
	data_Set = numpy.zeros((200,8))
	data_Set[:,0] = data.Species
	data_Set[:,1] = data.FrontalLip
	data_Set[:,2] = data.RearWidth
	data_Set[:,3] = data.Length
	data_Set[:,4] = data.Width
	data_Set[:,5] = data.Depth
	data_Set[:,6] = data.Male
	data_Set[:,7] = data.Female
	return data_Set


def classify(data):
	prediction = -(1/2)



data = load_data()
data_set = prepare_data(data)
numpy.save('data_set.npy',data_set)


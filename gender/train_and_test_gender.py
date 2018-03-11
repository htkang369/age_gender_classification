
import numpy as np
import sys
import pickle
import random
from net import Network

train_mode = 'gender'

def load_train_file(name):
	with open(name , 'rb') as f:
		return pickle.load(f)

def load_val_file(name):
	with open(name, 'rb') as f:
		return pickle.load(f)

def load_test_file(name):
	with open(name , 'rb') as f:
		return pickle.load(f)

def load_data(name):
	"""
	Load the whole data
	:param name:
	:return:
	"""
	with open(name, 'rb') as f:
		return pickle.load(f)

def train_model():
	"""
	This function will train model
	Tips: Load test,validation data first
	Then, seperately load training data, since training data is really huge.
	:return:
	"""
	path = '/home/htkang/bigdata/age_gender/data/wiki_new.pkl'
	data = load_data(path)
	## extract different type data
	train_dataset = data['train_dataset']/255
	#train_age_labels = data['train_age_labels']
	train_gender_labels = data['train_gender_labels']

	valid_dataset = data['valid_dataset']/255
	#valid_age_labels = data['valid_age_labels']
	valid_gender_labels = data['valid_gender_labels']

	test_dataset = data['test_dataset']/255
	#test_age_labels = data['test_age_labels']
	test_gender_labels = data['test_gender_labels']

	hight = 128
	channel = 1
	batch_size = 50
	learn_rate = 0.001
	n_output = 2
	total_size = train_dataset.shape[0]
	net = Network(
		n_output = n_output,
		n_length=hight,
		learning_rate=learn_rate,
		batch_size=batch_size,
		channel=channel,
		output_graph=False,
		use_ckpt=False
	)
	num_steps = 50000
	for i in range(num_steps):
		# randomly sample batch memory from all memory
		indices = np.random.permutation(total_size)[:batch_size]
		batch_x = train_dataset[indices, :, :, :]
		batch_y = train_gender_labels[indices, :]
		net.learn(batch_x,batch_y)
		if i%20==0:
			cost,accu_rate = net.get_accuracy_rate(batch_x,batch_y)
			print("Iteration: %i. Train loss %.5f, Minibatch accuracy:"" %.1f%%"
				  % (i, cost, accu_rate))
		if i%200==0:
			cost, accu_rate = net.get_accuracy_rate(valid_dataset,valid_gender_labels)
			print("Iteration: %i. Validation loss %.5f, Validation accuracy:"" %.1f%%"
				  % (i, cost, accu_rate))
			cost, accu_rate = net.get_accuracy_rate(test_dataset, test_gender_labels)
			print("Iteration: %i. Test loss %.5f, Test accuracy:"" %.1f%%"
				  % (i, cost, accu_rate))


def main():
	train_model()

if __name__=='__main__':
	main()	
	


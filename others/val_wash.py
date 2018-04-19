import cPickle as pickle
import numpy as np

def load_data(name):
	"""
	Load the whole data
	:param name:
	:return:
	"""
	with open(name, 'rb') as f:
		return pickle.load(f)

path = '/home/hengtong/project/age_gender/data/large/wiki_new.pkl'
data = load_data(path)

temp_train_dataset = data['train_dataset']
temp_valid_dataset = data['valid_dataset']
temp_test_dataset = data['test_dataset']

print('val train')
for i in range(len(temp_train_dataset)):
    if np.max(temp_train_dataset[i]) == 0.0:
        print i
        print ('Zero Alert!')

print('val val')
for i in range(len(temp_valid_dataset)):
    if np.max(temp_valid_dataset[i]) == 0.0:
        print i
        print ('Zero Alert!')

print('val test')
for i in range(len(temp_test_dataset)):
    if np.max(temp_test_dataset[i]) == 0.0:
        print i
        print ('Zero Alert!')


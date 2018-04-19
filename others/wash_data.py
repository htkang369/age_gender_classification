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

path = '/home/hengtong/project/age_gender/data/pkl_folder/wiki.pkl'
data = load_data(path)

## extract different type data
temp_train_dataset = data['train_dataset']
temp_train_age_labels = data['train_age_labels']
temp_train_gender_labels = data['train_gender_labels']

temp_valid_dataset = data['valid_dataset']
temp_valid_age_labels = data['valid_age_labels']
temp_valid_gender_labels = data['valid_gender_labels']

temp_test_dataset = data['test_dataset']
temp_test_age_labels = data['test_age_labels']
temp_test_gender_labels = data['test_gender_labels']

img_size = 128
img_depth = 1
train_dataset = np.ndarray((len(temp_train_dataset), img_size, img_size, img_depth), dtype=np.float32)
valid_dataset = np.ndarray((len(temp_valid_dataset), img_size, img_size, img_depth), dtype=np.float32)
test_dataset = np.ndarray((len(temp_test_dataset), img_size, img_size, img_depth), dtype=np.float32)

train_age_labels = np.ndarray((len(temp_train_age_labels), 4), dtype=np.float32)
valid_age_labels = np.ndarray((len(temp_valid_age_labels), 4), dtype=np.float32)
test_age_labels = np.ndarray((len(temp_test_age_labels), 4), dtype=np.float32)

train_gender_labels = np.ndarray((len(temp_train_gender_labels), 2), dtype=np.float32)
valid_gender_labels = np.ndarray((len(temp_valid_gender_labels), 2), dtype=np.float32)
test_gender_labels = np.ndarray((len(temp_test_gender_labels), 2), dtype=np.float32)

print('Processing Training Data')
k=0
for i in range(len(temp_train_dataset)):
    if np.max(temp_train_dataset[i])==0.0:
        print i
        continue
    else:
        train_dataset[k,:,:] = temp_train_dataset[i]
        train_age_labels[k,:] = temp_train_age_labels[i]
        train_gender_labels[k,:] = temp_train_gender_labels[i]
        k = k+1
train_dataset = train_dataset[0:k,:,:]
train_age_labels = train_age_labels[0:k,:]
train_gender_labels = train_gender_labels[0:k,:]


print('Processing Validation Data')
k=0
for i in range(len(temp_valid_dataset)):
    if np.max(temp_valid_dataset[i])==0.0:
        print i
        continue
    else:
        valid_dataset[k,:,:] = temp_valid_dataset[i]
        valid_age_labels[k, :] = temp_valid_age_labels[i]
        valid_gender_labels[k, :] = temp_valid_gender_labels[i]
        k = k+1
valid_dataset = valid_dataset[0:k,:,:]
valid_age_labels = valid_age_labels[0:k,:]
valid_gender_labels = valid_gender_labels[0:k,:]

print('Processing Testing Data')
k=0
for i in range(len(temp_test_dataset)):
    if np.max(temp_test_dataset[i])==0.0:
        print i
        continue
    else:
        test_dataset[k,:,:] = temp_test_dataset[i]
        test_age_labels[k, :] = temp_test_age_labels[i]
        test_gender_labels[k, :] = temp_test_gender_labels[i]
        k = k+1
test_dataset = test_dataset[0:k,:,:]
test_age_labels = test_age_labels[0:k,:]
test_gender_labels = test_gender_labels[0:k,:]

pickle_file = 'new_wiki.pkl'
try:
    f = open('/home/hengtong/project/age_gender/data/large/wiki_new.pkl', 'wb')
    save = {
    'train_dataset': train_dataset,
    'train_age_labels': train_age_labels,
    'train_gender_labels':train_gender_labels,
    'valid_dataset': valid_dataset,
    'valid_age_labels': valid_age_labels,
    'valid_gender_labels':valid_gender_labels,
    'test_dataset': test_dataset,
    'test_age_labels': test_age_labels,
    'test_gender_labels': test_gender_labels
    }
    pickle.dump(save, f, 2)
    f.close()
    print('Dumped successfully!')
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

print('Training set', train_dataset.shape, train_age_labels.shape,train_gender_labels.shape)
print('Validation set', valid_dataset.shape,valid_age_labels.shape,valid_gender_labels.shape)
print('Test set', test_dataset.shape, test_age_labels.shape,test_gender_labels.shape)
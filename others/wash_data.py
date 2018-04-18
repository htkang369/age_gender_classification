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

path = '/home/htkang/bigdata/age_gender/data/wiki.pkl'
data = load_data(path)

## extract different type data
temp_train_dataset = data['train_dataset']
train_age_labels = data['train_age_labels']
train_gender_labels = data['train_gender_labels']

temp_valid_dataset = data['valid_dataset']
valid_age_labels = data['valid_age_labels']
valid_gender_labels = data['valid_gender_labels']

temp_test_dataset = data['test_dataset']
test_age_labels = data['test_age_labels']
test_gender_labels = data['test_gender_labels']

img_size = 128
img_depth = 1
train_dataset = np.ndarray((len(temp_train_dataset), img_size, img_size, img_depth), dtype=np.float32)
valid_dataset = np.ndarray((len(temp_valid_dataset), img_size, img_size, img_depth), dtype=np.float32)
test_dataset = np.ndarray((len(temp_test_dataset), img_size, img_size, img_depth), dtype=np.float32)
print('Processing Training Data')
k=0
for i in range(len(temp_train_dataset)):
    if np.max(temp_train_dataset[i])==0.0:
        print i
        continue
    else:
        train_dataset[k,:,:] = temp_train_dataset[i]
        k = k+1
train_dataset = train_dataset[0:k,:,:]

print('Processing Validation Data')
k=0
for i in range(len(temp_valid_dataset)):
    if np.max(temp_valid_dataset[i])==0.0:
        print i
        continue
    else:
        valid_dataset[k,:,:] = temp_valid_dataset[i]
        k = k+1
valid_dataset = valid_dataset[0:k,:,:]

print('Processing Testing Data')
k=0
for i in range(len(temp_test_dataset)):
    if np.max(temp_test_dataset[i])==0.0:
        print i
        continue
    else:
        test_dataset[k,:,:] = temp_test_dataset[i]
        k = k+1
test_dataset = test_dataset[0:k,:,:]

pickle_file = 'washed_wiki.pkl'
try:
    f = open('/home/htkang/bigdata/age_gender/data/washed_wiki.pkl', 'wb')
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


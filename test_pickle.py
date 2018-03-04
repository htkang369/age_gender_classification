## This file is used as testing validation of pkl file

from six.moves import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt


#file=open('/home/htkang/bigdata/age_gender/data/imdb_data_test.pkl','rb')
#data = pickle.load
# data = np.ndarray((5,5))
# print data
# with open('/home/htkang/bigdata/age_gender/data/test.pkl', 'wb') as f:
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#     print("saved")

##todo the type of data
##data = {'image_inputs': np.array(real_imgs[slice_from:slice_to]),
##            'age_labels': np.array(age[slice_from:slice_to]),
##            'gender_labels': np.array(gender[slice_from:slice_to])
##            }
# ##
# save = {
#     'train_dataset': train_dataset,
#     'train_age_labels': train_age_labels,
#     'train_gender_labels':train_gender_labels,
#     'valid_dataset': valid_dataset,
#     'valid_age_labels': valid_age_labels,
#     'valid_gender_labels':valid_gender_labels,
#     'test_dataset': test_dataset,
#     'test_age_labels': test_age_labels,
#     'test_gender_labels': test_gender_labels
#     }

path = '/home/htkang/bigdata/age_gender/data/wiki_new.pkl'
with open(path,'rb') as file:
    data = pickle.load(file)
    valid_dataset = data['valid_dataset']
print valid_dataset.shape
print valid_dataset[4,:,:]/255
im = np.reshape(valid_dataset[4,:,:],(128,128))
plt.imshow(im)
plt.show()

def wash_image(path):
    """
    This function is used as removing dirty data
    :param path:
    :return:
    """
    with open(path,'rb') as file:
        data = pickle.load(file)
        train_dataset = data['train_dataset']
        train_age_labels = data['train_age_labels']
        train_gender_labels = data['train_gender_labels']

        valid_dataset = data['valid_dataset']
        valid_age_labels = data['valid_age_labels']
        valid_gender_labels =  data['valid_gender_labels']

        test_dataset = data['test_dataset']
        test_age_labels = data['test_age_labels']
        test_gender_labels =  data['test_gender_labels']

    new_valid_dataset = []
    new_valid_age_labels =[]
    new_valid_gender_labels = []

    new_train_dataset = []
    new_train_age_labels =[]
    new_train_gender_labels = []

    new_test_dataset = []
    new_test_age_labels =[]
    new_test_gender_labels = []
    print('valid check')
    k = 0
    for i in range(len(valid_dataset)):
        if np.max(np.max(valid_dataset[i]))==0.0:
            print i
            continue
        new_valid_dataset.append(valid_dataset[i,:,:])
        new_valid_age_labels.append(valid_age_labels[i,:])
        new_valid_gender_labels.append(valid_gender_labels[i,:])
        k += 1

    print('train check')
    k = 0
    for i in range(len(train_dataset)):
        if np.max(np.max(train_dataset[i]))==0.0:
            print i
            continue
        new_train_dataset.append(train_dataset[i,:,:])
        new_train_age_labels.append(train_age_labels[i,:])
        new_train_gender_labels.append(train_gender_labels[i,:])
        k += 1

    print('test check')

    k = 0
    for i in range(len(test_dataset)):
        if np.max(np.max(test_dataset[i]))==0.0:
            print i
            continue
        new_test_dataset.append(test_dataset[i,:,:])
        new_test_age_labels.append(test_age_labels[i,:])
        new_test_gender_labels.append(test_gender_labels[i,:])
        k +=1

    with open('/home/htkang/bigdata/age_gender/data/wiki_new.pkl','wb') as new_file:
        save = {
            'train_dataset': np.array(new_train_dataset),
            'train_age_labels': np.array(new_train_age_labels),
            'train_gender_labels': np.array(new_train_gender_labels),
            'valid_dataset': np.array(new_valid_dataset),
            'valid_age_labels': np.array(new_valid_age_labels),
            'valid_gender_labels': np.array(new_valid_gender_labels),
            'test_dataset': np.array(new_test_dataset),
            'test_age_labels': np.array(new_test_age_labels),
            'test_gender_labels': np.array(new_test_gender_labels)
        }
        pickle.dump(save, new_file, 2)
    new_file.close()
    print ('saved successfully')


#path = '/home/htkang/bigdata/age_gender/data/wiki.pkl'
#wash_image(path)

# coding: utf-8

# # IMDB-WIKI
# ##  Multi-task age and gender classification

# On the original paper [DEX: Deep EXpectation of apparent age from a single image](https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_01229.pdf) the authors were able to display remarkable results in classifying the age of an individual based on a given image alone. 
# 
# Let see how accuracy (bad I guess), with limited resources, we can get with self-construct architecture. And not only age, we also classifying gender by using multi-task training technique.

# In[1]:

import os
from six.moves import cPickle as pickle
import numpy as np
import scipy.io as sio
import scipy.misc as spm
from scipy import ndimage
import datetime
import matplotlib.image as plt
from IPython.display import Image, display
from skimage.transform import resize

#IMG_DIR = r'/home/ubuntu/coding/cnn/datasets/imdb_crop'
#MAT_FILE = r'/home/ubuntu/coding/cnn/datasets/imdb_crop/imdb.mat'
IMG_DIR = '/home/hengtong/project/age_gender/data/wiki_crop'
MAT_FILE = '/home/hengtong/project/age_gender/data/wiki_crop/wiki.mat'
img_depth = 1
img_size = 128
num_classes = 2
PROTOCOL = 2

max_bytes = 2**31 - 1
max_num = 40000 # number of image


# First, the labels, which was not easily obtained. The meta data is stored separately and in a .mat file. (Yes, matlab)!
# 
# The age parameter, requires us to calculate by taking the ```photo_taken``` and subtracting the ```dob```, the date of birth. Sounds easy? No ... as the dob is stored as a Matlab serial number.
# 
# Luckily we can use the ```scipy.io.loadmat``` to load the ```.mat``` file to python accessible (kind of) format. We can access the ```dob``` by some proper indexing, and convert the Matlab serial number to a usable format by using ```datetime.date.fromordinal( serial_number ).year```.

# In[2]:

def reformat_date(mat_date):
    dt = datetime.date.fromordinal(np.max([mat_date - 366, 1])).year
    return dt


# In[3]:

def create_path(path):
    return os.path.join(IMG_DIR, path[0])


# In[4]:

mat_struct = sio.loadmat(MAT_FILE)
data_set = [data[0] for data in mat_struct['wiki'][0, 0]]

keys = ['dob',
    'photo_taken',
    'full_path',
    'gender',
    'name',
    'face_location',
    'face_score',
    'second_face_score',
    'celeb_names',
    'celeb_id'
]

imdb_dict = dict(zip(keys, np.asarray(data_set)))
imdb_dict['dob'] = [reformat_date(dob) for dob in imdb_dict['dob']]
imdb_dict['full_path'] = [create_path(path) for path in imdb_dict['full_path']]

# Add 'age' key to the dictionary
imdb_dict['age'] = imdb_dict['photo_taken'] - imdb_dict['dob']

print("Dictionary created...")


# The IMDB dataset has total 460,723 face images from 20,284 celebrities. 
# 
# We will ignore:
# * images with more than one face
# * gender is NaN
# * invalid age.

# In[5]:


raw_path = imdb_dict['full_path'][0:max_num ]
raw_age = imdb_dict['age'][0:max_num ]
raw_gender = imdb_dict['gender'][0:max_num ]
raw_sface = imdb_dict['second_face_score'][0:max_num ]

age = []
gender = []
imgs = []
for i, sface in enumerate(raw_sface):
    if i%2==0:
        print("Processing {0} of {1}".format(i,len(raw_sface)))
#         display(Image(filename=raw_path[i]))
        print("Second face score: {}".format(sface), end=" ")
        print("Age: {}".format(raw_age[i]), end=" ")
        print("Gender: {}".format(raw_gender[i]))
    if np.isnan(sface) and raw_age[i] >= 0 and not np.isnan(raw_gender[i]):
            age.append(raw_age[i])
            gender.append(raw_gender[i])
            imgs.append(raw_path[i])
            print (raw_path[i])


# Since some photos are colored and some are gray scale, while the sizes are not consistent. Moreover, processing file in RBG format is too big, when attemp to save objects to pickle file, 60000 file is equivalent to 11GB. So we gonna resize image to 128x128, convert to grayscale.
# 
# Also due to limit of resources and time, I only pick first ```100000``` images to train.

# In[6]:

# Convert images path to images.

# only take a subset of dataset: first 10000 imgs
# dataset = np.ndarray(shape=(100000, img_size, img_size, img_depth), dtype=np.float32)
import matplotlib.pyplot as plt
if os.path.exists(os.getcwd()+"/pkl_folder/imdb_data_train.pkl") and os.path.exists(
    os.getcwd()+"/pkl_folder/imdb_data_valid.pkl") and os.path.exists(
    os.getcwd()+"/pkl_folder/imdb_data_test.pkl"):
    print("Dataset already present - Skip convert images to images.")
else:
    print("Converting images path to images.")
    real_imgs = []
    tmp = []
    for i, img_path in enumerate(imgs):
        if i==100000:
            break
        im = spm.imread(img_path,flatten=1) #flatten==0,rgb,==1,gray
        
        #plt.imshow(im)
        #plt.show()
        #print (im.shape)
        tmp = np.asarray(spm.imresize(im, (128, 128)), dtype=np.float32)
        #print (tmp.shape)
        #plt.imshow(tmp)
        #plt.show()
        real_imgs.append(tmp)
        print (i)

    print("Original size: {0} - Preprocess size: {1}".format(len(raw_sface), len(real_imgs)))
    
#     print("Converting images path to images.")
#     for i, img_path in enumerate(imgs):
#         if i == 100000:
#             break
#         image_data = resize(((ndimage.imread(img_path).astype(float) - img_depth / 2) / img_depth), 
#                             (img_size, img_size, img_depth), mode='reflect')
#         dataset[i, :, :, :] = image_data

#     print("Original size: {0} - Preprocess size: {1}".format(len(raw_sface), len(dataset)))


# Dump 3 datasets to pickles.

# In[8]:

def dump_data(file_path, slice_from, slice_to):
    data = {'image_inputs': np.array(real_imgs[slice_from:slice_to]),
            'age_labels': np.array(age[slice_from:slice_to]),
            'gender_labels': np.array(gender[slice_from:slice_to])
            }
    print("Dataset dump size: {}".format(len(data['image_inputs'])))
    with open(file_path,'wb') as f:
        pickle.dump(data, f, PROTOCOL)
    print("Dumped to {}".format(file_path))

def create_pickle(ra_num):
    data_train_path = "/home/hengtong/project/age_gender/data/pkl_folder/imdb_data_train.pkl"
    data_valid_path = "/home/hengtong/project/age_gender/data/pkl_folder/imdb_data_valid.pkl"
    data_test_path = "/home/hengtong/project/age_gender/data/pkl_folder/imdb_data_test.pkl"
    if os.path.exists(data_train_path) and os.path.exists(
        data_valid_path) and os.path.exists(
        data_test_path):
        # You may override by setting force=True.
        print("Dataset already present - Skip pickling.")
        
    else:
        dump_data(data_train_path, 0, ra_num[0])
        dump_data(data_valid_path, ra_num[0], ra_num[0]+ra_num[1])
        dump_data(data_test_path, ra_num[0]+ra_num[1], ra_num[0]+ra_num[1]+ra_num[2])

    return data_train_path, data_valid_path, data_test_path
# print (os.getcwd())
total_nu = len(real_imgs)
ra = [0.85,0.05,0.1]
ra_num = [int(total_nu*ra[0]),int(total_nu*ra[1]),total_nu-int(total_nu*ra[0])-int(total_nu*ra[1])]
print (ra_num)
data_train_path, data_valid_path, data_test_path = create_pickle(ra_num)


# As we are using only a subset of the data, and also using a self-constructed model that has a much smaller capacity, thus we need to take steps to adjust accordingly.
# 
# The original paper uses 101101 age classes, which was appropriate for the their data set size and learning architecture. As we are only using a small subset of the data and a very simple model, the number of classes was set to 4:
# * Young    (30yrs < age)
# * Middle   (30 <= age <45)
# * Old      (45 <= age < 60)
# * Very Old (60 <= age)

# In[9]:

def convert_label(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            data_train = pickle.load(f)
            labels = np.ndarray((len(data_train['image_inputs']), num_classes), dtype=np.int32)
            dataset = np.ndarray((len(data_train['image_inputs']), img_size, img_size, img_depth), dtype=np.float32)
            # let's shuffle to have random dataset
            np.random.shuffle(dataset)
            dataset = data_train['image_inputs']
            for i, age_label in enumerate(data_train['age_labels']):
                if i==len(data_train['image_inputs']):
                    break
                if age_label < 30:
                    age = 0
                elif age_label <= 45:
                    age = 1
                elif age_label < 60:
                    age = 2
                elif age_label >= 60:
                    age = 3
                else:
                    continue
                labels[i,:] = np.array([age, data_train['gender_labels'][i]])
            return dataset, labels
            
    except Exception as e:
        print('Unable to process data from', pickle_file, ':', e)
        raise

train_dataset, train_labels = convert_label(data_train_path)
valid_dataset, valid_labels = convert_label(data_valid_path)
test_dataset, test_labels = convert_label(data_test_path)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

# In[10]:

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


# Reformat into a TensorFlow-friendly shape:
# * convolutions need the image data formatted as a cube (width by height by channels)
# * labels as float 1-hot encodings.

# In[11]:

num_channels = img_depth # = 1 (Grayscale)

def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, img_size, img_size, num_channels)).astype(np.float32)
    one_hot_age_labels = np.ndarray((len(labels),4), dtype=np.float32)
    one_hot_gender_labels = np.ndarray((len(labels),2), dtype=np.float32)
    for i, label in enumerate(labels):
        one_hot_age = (np.arange(4)==label[0]).astype(np.float32)
        one_hot_gender = (np.arange(2)==label[1]).astype(np.float32)
        one_hot_age_labels[i,:] = np.array(one_hot_age)
        one_hot_gender_labels[i,:] = np.array(one_hot_gender)
        print ('age:',format(label[0]))
        print (one_hot_age_labels[i,:])
        print ('gender',format(label[1]))
        print (one_hot_gender_labels[i,:])
#     labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, one_hot_age_labels,one_hot_gender_labels

train_dataset, train_age_labels ,train_gender_labels= reformat(train_dataset, train_labels)
valid_dataset, valid_age_labels,valid_gender_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_age_labels,test_gender_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_age_labels.shape,train_gender_labels.shape)
print('Validation set', valid_dataset.shape,valid_age_labels.shape,valid_gender_labels.shape)
print('Test set', test_dataset.shape, test_age_labels.shape,test_gender_labels.shape)


# In[12]:

print ((np.arange(2)==1.0).astype(np.float32))
print ((np.arange(4)==3).astype(np.float32))


# Save to final pickle file.

# In[13]:

pickle_file = 'wiki.pkl'
try:
    f = open('/home/hengtong/project/age_gender/data/pkl_folder/wiki.pkl', 'wb')
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
    pickle.dump(save, f, PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat("/home/hengtong/project/age_gender/data/pkl_folder/wiki.pkl")
print('Compressed pickle size:', statinfo.st_size)







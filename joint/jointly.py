
import numpy as np
import sys
import pickle
import random
from net import Network


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
    train_age_labels = data['train_age_labels']
    train_gender_labels = data['train_gender_labels']

    valid_dataset = data['valid_dataset']/255
    valid_age_labels = data['valid_age_labels']
    valid_gender_labels = data['valid_gender_labels']

    test_dataset = data['test_dataset']/255
    test_age_labels = data['test_age_labels']
    test_gender_labels = data['test_gender_labels']

    hight = 128
    channel = 1
    batch_size = 50
    learn_rate = 0.001
    n_output_age = 4
    n_output_gen = 2
    total_size = train_dataset.shape[0]

    net = Network(
        n_output_gen = n_output_gen,
        n_output_age = n_output_age,
        n_length=hight,
        learning_rate=learn_rate,
        batch_size=batch_size,
        channel=channel,
        output_graph=True,
        use_ckpt=False
    )
    epoch = 4000
    iteration = int(total_size / batch_size)
    i=1 # total training time
    for e in range(epoch):
        # randomly sample batch memory from all memory
        indices = np.random.permutation(total_size)
        for ite in range(iteration):
            mini_indices = indices[ite*batch_size:(ite+1)*batch_size]
            batch_x = train_dataset[mini_indices, :, :, :]
            batch_y_gen = train_gender_labels[mini_indices, :]
            batch_y_age = train_age_labels[mini_indices, :]
            net.learn(batch_x,batch_y_gen,batch_y_age)
            if i%20==0:
                cost,rate_gen,rate_age = net.get_accuracy_rate(batch_x,batch_y_gen,batch_y_age)
                print("Iteration: %i. Train loss %.5f, Minibatch gen accuracy:"" %.1f%%,Minibatch age accuracy:"" %.1f%%"
                      % (i, cost, rate_gen,rate_age))
            if i%50==0:
                cost, rate_gen,rate_age = net.get_accuracy_rate(valid_dataset,valid_gender_labels,valid_age_labels)
                print("Iteration: %i. Validation loss %.5f, Validation gen accuracy:" " %.1f%% ,Validation age accuracy:" " %.1f%%"
                      % (i, cost, rate_gen,rate_age))
                cost, accu_rate_gen,accu_rate_age = net.get_accuracy_rate(test_dataset, test_gender_labels,test_age_labels)
                print("Iteration: %i. Test loss %.5f, Test gen accuracy:"" %.1f%%,Test age accuracy:"" %.1f%%"
                      % (i, cost, accu_rate_gen,accu_rate_age))
            if i%500==0:
                net.save_parameters()

            if i%5==0: # save histogram
                net.merge_hist(batch_x,batch_y_gen,batch_y_age)
            i = i+1





def main():
    train_model()

if __name__=='__main__':
    main()
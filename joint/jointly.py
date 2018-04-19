
import numpy as np
import sys
import pickle
import random
from net import Network
import matplotlib.pyplot as plt


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
    path = '/home/hengtong/project/age_gender/data/small/wiki_new.pkl'
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
    batch_size = 128
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
    epoch = 10
    iteration = int(total_size / batch_size)

    early_stop = 0 #flag of early stopping
    i=1 # total training time
    accu_train_gen=[]
    accu_valid_gen=[]
    accu_test_gen=[]
    accu_train_age = []
    accu_valid_age = []
    accu_test_age = []
    train_rate_gen,train_rate_age = 0,0

    for e in range(epoch):
        print("-------------------------------")
        print("epoch %d" %(e+1))
        # randomly sample batch memory from all memory
        indices = np.random.permutation(total_size)
        for ite in range(iteration):
            mini_indices = indices[ite*batch_size:(ite+1)*batch_size]
            batch_x = train_dataset[mini_indices, :, :, :]
            batch_y_gen = train_gender_labels[mini_indices, :]
            batch_y_age = train_age_labels[mini_indices, :]
            net.learn(batch_x,batch_y_gen,batch_y_age)

            if i%50==0:
                cost,train_rate_gen,train_rate_age = net.get_accuracy_rate(batch_x,batch_y_gen,batch_y_age)
                print("Iteration: %i. Train loss %.5f, Minibatch gen accuracy:"" %.1f%%,Minibatch age accuracy:"" %.1f%%"
                      % (i, cost, train_rate_gen,train_rate_age))
                accu_train_gen.append(train_rate_gen),accu_train_age.append(train_rate_age)

            if i%50==0:
                cost, valid_rate_gen,valid_rate_age = net.get_accuracy_rate(valid_dataset,valid_gender_labels,valid_age_labels)
                print("Iteration: %i. Validation loss %.5f, Validation gen accuracy:" " %.1f%% ,Validation age accuracy:" " %.1f%%"
                      % (i, cost, valid_rate_gen,valid_rate_age))
                accu_valid_gen.append(valid_rate_gen), accu_valid_age.append(valid_rate_age)
                cost, test_rate_gen,test_rate_age = net.get_accuracy_rate(test_dataset, test_gender_labels,test_age_labels)
                print("Iteration: %i. Test loss %.5f, Test gen accuracy:"" %.1f%%,Test age accuracy:"" %.1f%%"
                      % (i, cost, test_rate_gen,test_rate_age))
                accu_test_gen.append(test_rate_gen), accu_test_age.append(test_rate_age)
            if i%500==0:
                net.save_parameters()

            if i%5==0: # save histogram
                net.merge_hist(batch_x,batch_y_gen,batch_y_age)
            i = i+1

        # early stopping
        if train_rate_gen==100 and train_rate_age==100:
            if early_stop==10:
                print("Early Stopping!")
                break
            else:early_stop = early_stop+1

    net.plot_cost() # plot trainingi cost

    plt.figure()   # plot accuracy
    plt.plot(np.arange(len(accu_train_gen)), accu_train_gen,label='train gender',linestyle='--' )
    plt.plot(np.arange(len(accu_valid_gen)), accu_valid_gen,label='valid gender',linestyle='-')
    plt.plot(np.arange(len(accu_test_gen)), accu_test_gen,label='test gender',linestyle=':')
    plt.ylabel('gender accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('gender.png')

    plt.figure()  # plot accuracy
    plt.plot(np.arange(len(accu_train_age)), accu_train_age,label='train age',linestyle='--')
    plt.plot(np.arange(len(accu_valid_age)), accu_valid_age,label='valid age',linestyle='-')
    plt.plot(np.arange(len(accu_test_age)), accu_test_age,label='test age',linestyle=':')
    plt.ylabel('age accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('age.png')
    # plt.show()


def main():
    train_model()

if __name__=='__main__':
    main()
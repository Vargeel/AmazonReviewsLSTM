import six.moves.cPickle as pickle
import numpy as np

def prepare_data(reviews, labels):
    reviews = reviews[0]
    lengths = [len(s) for s in reviews]
    n_reviews = len(reviews)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_reviews)).astype('int64')
    x_mask = np.zeros((maxlen, n_reviews)).astype('float32')
    labels = np.array(labels).astype('int32')
    for idx, review in enumerate(reviews):
        x[:lengths[idx], idx] = review
        x_mask[:lengths[idx], idx] = 1. # adding blank space to fill it to the max length

    return np.transpose(x,(1,0)), np.transpose(x_mask,(1,0)), labels[0]


def load_data(path="data.pkl", training_to_validation_ratio=0.8):

    f = open(path, 'rb')

    data_set = pickle.load(f)
    f.close()

    # split training set into validation set

    data_set_x, data_set_y = data_set
    n_reviews = len(data_set_x)
    sidx = np.random.permutation(n_reviews)
    data_set_x = np.asarray(data_set_x)[sidx]
    data_set_y = np.asarray(data_set_y)[sidx]
    n_training = int(n_reviews * (training_to_validation_ratio))
    training_set_x = [data_set_x[:n_training]]
    training_set_y = [data_set_y[:n_training]]
    validation_set_x = [data_set_x[n_training:]]
    validation_set_y = [data_set_y[n_training:]]

    training_set = (training_set_x, training_set_y)
    validation_set = (validation_set_x, validation_set_y)




    return training_set, validation_set
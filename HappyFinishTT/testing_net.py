import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import scripts
from training_net import networkConfig
from preprocess import preprocess
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '#' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()



def get_config():
    return networkConfig

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial,name=name)

def getBatch(mode,bsize,data_train,mask_train,labels_train,data_validation,mask_validation,labels_validation):

    batches = []

    if mode == 'train':
        training_set_size = labels_train.shape[0]
        rr = np.random.randint(training_set_size,size=(training_set_size))
        n_batches = int(np.floor(np.divide(training_set_size,float(bsize))))
        for batch_idx in range(n_batches):
            batch = [0]*3
            batch[0] = np.zeros((bsize,networkConfig['num_steps']))
            batch[1] = np.zeros((bsize,networkConfig['num_steps']))
            batch[2] = np.zeros((bsize,networkConfig['output_size']))
            batch[0] = data_train[rr[batch_idx*bsize:(batch_idx+1)*bsize],:networkConfig['num_steps']]
            batch[1] = mask_train[rr[batch_idx*bsize:(batch_idx+1)*bsize],:networkConfig['num_steps']]
            batch[2] = labels_train[rr[batch_idx*bsize:(batch_idx+1)*bsize],:]
            batches.append(batch)

    if mode == 'test':
        validation_set_size = labels_validation.shape[0]
        r = np.random.randint(validation_set_size,size=(validation_set_size))
        n_batches_test = int(np.floor(np.divide(validation_set_size,float(bsize))))
        for batch_idx_test in range(n_batches_test):
            batch = [0]*3
            batch[0] = np.zeros((bsize,networkConfig['num_steps']))
            batch[1] = np.zeros((bsize,networkConfig['num_steps']))
            batch[2] = np.zeros((bsize,networkConfig['output_size']))
            batch[0] = data_validation[r[batch_idx_test*bsize:(batch_idx_test+1)*bsize],:networkConfig['num_steps']]
            batch[1] = mask_validation[r[batch_idx_test*bsize:(batch_idx_test+1)*bsize],:networkConfig['num_steps']]
            batch[2] = labels_validation[r[batch_idx_test*bsize:(batch_idx_test+1)*bsize],:]
            batches.append(batch)

    return batches






def test(path_preproc):
    print('Starting preprocess')
    preprocess(path_preproc,test=True)
    print('Finished preprocess')
    path = 'data_test.pkl'
    print('Loading data')
    test, _ = scripts.load_data(path=path, training_to_validation_ratio=1.0)
    print('Data loaded')
    test_data, test_mask, test_labels = scripts.prepare_data(test[0], test[1])
    print('Training data shape : data, mask, label ')
    print(test_data.shape, test_mask.shape, test_labels.shape)

    config = get_config()

    sess = tf.InteractiveSession()

    batch_size = config['batch_size']
    num_steps = config['num_steps']
    output_size = config['output_size']

    input_data = tf.placeholder(tf.float32, [batch_size, num_steps], name="inputs")
    mask = tf.placeholder(tf.float32, [batch_size, num_steps], name="mask")
    labels = tf.placeholder(tf.float32, [batch_size, output_size], name="labels")
    input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob"),
    output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob"),

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps)
    # Required shape: 'n_steps' tensors list of shape (batch_size)
    input_data_reshaped = tf.reshape(input_data,[batch_size,num_steps,1])

    # Permuting batch_size and n_steps
    x = tf.transpose(input_data_reshaped, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, 1])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    inputs = tf.split(x, num_steps, 0)

    #add LSTM cell and dropout nodes
    cell = rnn.BasicLSTMCell(config['hidden_size'], forget_bias=1.0)

    cell = rnn.DropoutWrapper(cell,input_keep_prob=input_keep_prob,output_keep_prob=output_keep_prob)

    cell_fw = rnn.BasicLSTMCell(config['hidden_size'], forget_bias=1.0)
    cell_bw = rnn.BasicLSTMCell(config['hidden_size'], forget_bias=1.0)


    # Get lstm cell output
    if config['bidirectional'] == False:
        outputs, states = rnn.static_rnn(cell=cell, inputs=inputs, dtype=tf.float32)
    else:
        outputs, states, _ = rnn.static_bidirectional_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs, dtype=tf.float32)
    if config['Masked']:
        outputs_reduced = []
        mask_split = tf.split(tf.reshape(mask,[num_steps,batch_size]), num_steps)
    # we kill any output that was generated by empty data
        for idx in range(num_steps):
            mm = tf.transpose(mask_split[idx])
            oo = outputs[idx]
            if config['bidirectional'] == False:
                mm2 = tf.tile(mm,(1,config['hidden_size']))
            else:
                mm2 = tf.tile(mm,(1,config['hidden_size'] * 2))

            masked_output = tf.multiply(oo,mm2)
            outputs_reduced.append(masked_output)

    else:
        outputs_reduced = outputs

    # Linear activation, using rnn inner loop last output
    if config['bidirectional'] == False:
        W1 = weight_variable([config['hidden_size'],config['output_size']],'W1')
        b1 = bias_variable([config['output_size']],'b1')
    else:
        W1 = weight_variable([2*config['hidden_size'],config['output_size']],'W1')
        b1 = bias_variable([config['output_size']],'b1')
    prediction = tf.matmul(outputs_reduced[-1], W1) + b1

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    cost = tf.reduce_sum(tf.square(prediction - labels))

    learningRate = config['learning_rate']
    trainStep = tf.train.AdamOptimizer(learningRate).minimize(cost)
    # correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    error = tf.reduce_mean(tf.sqrt(tf.square(tf.rint(prediction) - labels)))



    saver = tf.train.Saver()
    saver.restore(sess, "models/model.ckpt")



    batches = getBatch('test',batch_size,None,None,None,test_data,test_mask,test_labels)
    batch_count = 0
    error_train = 0
    error_test = 0

    for batch in batches :
        batch_count+=1
        print_progress(batch_count, len(batches))
        error_train += error.eval(feed_dict={input_data: batch[0], mask: batch[1], labels: batch[2], input_keep_prob: 1.0, output_keep_prob: 1.0})
    print("Test error: {}".format(error_train/len(batches)))



                
    print('\n')


if __name__ == '__main__':
    test('data.pkl')

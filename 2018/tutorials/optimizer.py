import numpy as np

import tensorflow as tf

class BatchReader(object):
    
    def __init__(self,
                 data_dict,
                 batch_size,
                 shuffle=True,
                 shuffle_seed=0,
                 pad=True,
                 processors=None):
        self.data_dict = data_dict        
        self.batch_size = batch_size
        _k = data_dict.keys()[0]
        self.data_length = data_dict[_k].shape[0]
        self.total_batches = (self.data_length - 1) // self.batch_size + 1
        self.curr_batch_num = 0
        self.curr_epoch = 1
        self.pad = pad
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed        
        if self.shuffle:
            self.rng = np.random.RandomState(seed=self.shuffle_seed)
            self.perm = self.rng.permutation(self.data_length)
        if processors is None:
            processors = {}
        self.processors = processors
        
    def __iter__(self):
        return self

    def next(self):
        return self.get_next_batch()
        
    def get_next_batch(self):
        data = self.get_batch(self.curr_batch_num)
        self.increment_batch_num()
        return data

    def increment_batch_num(self):
        m = self.total_batches
        if (self.curr_batch_num >= m - 1):
            self.curr_epoch += 1
            if self.shuffle:
                self.perm = self.rng.permutation(self.data_length)
        self.curr_batch_num = (self.curr_batch_num + 1) % m

    def get_batch(self, cbn):
        data = {}
        startv = cbn * self.batch_size
        endv = (cbn + 1) * self.batch_size
        if self.pad and endv > self.data_length:
            startv = self.data_length - self.batch_size
            endv = startv + self.batch_size
        elif endv > self.data_length:
            endv = min(endv, self.data_length)
        for k in self.data_dict:
            if self.shuffle:
                batch_inds = np.sort(self.perm[startv: endv])
            else:
            	batch_inds = range(startv, endv)
            data[k] = self.get_data(k, batch_inds)
        return data

    def get_data(self, k, inds):
        if k in self.processors:
            processor = self.processors[k]
            return processor(self.data_dict[k], inds)
        else:
            return self.data_dict[k][inds]

    

class TF_Optimizer(object):
    """Make the tensorflow SGD-style optimizer into a scikit-learn compatible class
       Uses BatchReader for stochastically getting data batches.
       
       model_func: function which returns tensorflow nodes for
                     predictions, data_input
        
       loss_func: function which takes model_func prediction output node and 
                  returns tensorflow nodes for
                     loss, label_input
                     
       optimizer_class: which tensorflow optimizer class to when learning the model parameters
       
       batch_size: which batch size to use in training
       
       train_iterations: how many iterations to run the optimizer for 
           --> this should really be picked automatically by like when the training
               error plateaus
               
        model_kwargs:  dictionary of additional arguments for the model_func
        
        loss_kwargs: dictionary of additional arguments for the loss_func
        
        optimizer_args, optimizer_kwargs: additional position and keyword args for the
         optimizer class
         
        sess: tf session to use (will be constructed if not passed) 
        
        train_shuffle: whether to shuffle example order during training
       
    """ 
    
    def __init__(self, 
                 model_func,
                 loss_func, 
                 optimizer_class,
                 train_iterations,
                 batch_size=None,
                 model_kwargs=None,
                 loss_kwargs=None,
                 optimizer_args=(),
                 optimizer_kwargs=None,
                 sess=None,
                 train_shuffle=False,
                 data_processors=None
                 ):
                        
        self.model_func = model_func
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        self.loss_func = loss_func
        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss_kwargs = loss_kwargs
        self.train_shuffle=train_shuffle
        
        self.train_iterations = train_iterations
        self.batch_size = batch_size
        if data_processors is None:
            data_processors = {}
        self.data_processors = data_processors
     
        if sess is None:
            sess = tf.Session()
        self.sess = sess
                
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = optimizer_class(*optimizer_args, 
                                         **optimizer_kwargs)

    def get_data(self, data, labels, shuffle, shuffle_seed, pad):
        if 'data' in self.data_processors:
            td = self.data_processors['data'](data,
                                              range(self.batch_size))
        else:
            td = data[:1]
        fshape = td.shape[1:]
        if not hasattr(self, 'data_holder'):
            self.data_holder = tf.placeholder(shape=(None,) + fshape,
                                     dtype=tf.float32,
                                     name='data')

        data_dict = {self.data_holder: data}            
        if labels is not None:
            if 'labels' in self.data_processors:
                tl = self.data_processors['labels'](labels,
                                                range(self.batch_size))
            else:
                tl = labels[:1]
            lshape = tl.shape[1:]
            if not hasattr(self, 'label_holder'):
                self.label_holder = tf.placeholder(shape=(None,) + lshape,
                                     dtype=tf.float32,
                                     name='labels')
            data_dict[self.label_holder] = labels
 

        data_processors={}
        if self.data_processors:
            if 'data' in self.data_processors:
                data_processors[self.data_holder] = self.data_processors['data']
            if labels is not None and 'labels' in self.data_processors:
                data_processors[self.label_holder] = self.data_processors['labels']
        data_reader = BatchReader(data_dict=data_dict,
                                  batch_size=self.batch_size,
                                  shuffle=shuffle,
                                  shuffle_seed=shuffle_seed,
                                  pad=pad,
                                  processors=data_processors,
                                  **kwargs)
        return data_reader

        
    def fit(self, train_data, train_labels, report_freq=100):
        if not isinstance(train_data, tf.Tensor):
            data_reader = self.get_data(train_data,
                                        train_labels,
                                        shuffle=self.train_shuffle,
                                        shuffle_seed=0,
                                        pad=True)
        else:
            data_reader = None
            self.data_holder = train_data
            self.label_holder = train_labels
            
        self.model = self.model_func(self.data_holder,
        			     self.label_holder,
        			     **self.model_kwargs)
        self.loss = self.loss_func(self.model,
                                   self.label_holder,
                                   **self.loss_kwargs)

        scope = tf.get_variable_scope()
        optim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        self.optimizer_op = self.optimizer.minimize(self.loss, var_list=optim_vars)
        optim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)         
        init_op = tf.variables_initializer(optim_vars)
    
        self.sess.run(init_op)
        
        self.losses = []                      
        for i in range(self.train_iterations):
            if data_reader is not None:
                data_batch = data_reader.next()
                output = self.sess.run({'opt': self.optimizer_op,
                                        'loss': self.loss}, 
                                        feed_dict=data_batch)
            else:
                output = self.sess.run({'opt': self.optimizer_op,
                                        'loss': self.loss})
            if i % report_freq == 0:
                print('iteration %d loss %.3f' % (i, output['loss']))
            self.losses.append(output['loss'])
                
    def predict(self, test_data, test_labels=None, num_batches=None):
        label_output = None
        if not isinstance(test_data, tf.Tensor):
            data_reader = self.get_data(test_data,
                                        None,
                                        shuffle=False,
                                        shuffle_seed=0,
                                        pad=False)
        else:
            label_output = []
            data_reader = None
        preds = []
        if num_batches is None:
            num_batches = test_data.total_batches

        
        for i in range(num_batches):
            if data_reader is not None:
                data_batch = test_data.get_batch(i)
                pred_batch = self.sess.run(self.model, feed_dict=data_batch)
            else:
                model = self.model_func(test_data,
                                        test_labels,
                                        **self.model_kwargs)

                pred_batch, labels_batch = self.sess.run([model, test_labels])
                label_output.append(labels_batch)
            preds.append(pred_batch)

        if label_output is None:
            return np.row_stack(preds)
        else:
            return np.row_stack(preds), np.row_stack(label_output)


def binarize_labels(labels):
    """takes discrete-valued labels and binarizes them into {-1, 1}-value format
       returns:
           binarized_labels: of shape (num_stimuli, num_categories)
           unique_labels: actual labels indicating order of first axis in binarized_labels
    """
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    binarized_labels = np.array([2 * (labels == c) - 1 for 
                                    c in unique_labels]).T.astype(int)
    return binarized_labels, unique_labels
    

class TF_OVA_Classifier(TF_Optimizer):
    """
    Subclass of TFOptimizer for use with categorizers. Basically, this class 
    handles data binarization (in the fit method) and un-binarization 
    (in the predict method), so that we can use the class with the function:
    
         train_and_test_scikit_classifier
    
    that we've previously defined. 
    
    The predict method here implements a one-vs-all approach for multi-class problems.
    """
    
    def fit(self, train_data, train_labels):
        #binarize labels
        num_features = train_data.shape[1]
        binarized_labels, classes_ = binarize_labels(train_labels)
        #set .classes_ attribute, since this is needed by train_and_test_scikit_classifier
        self.classes_ = classes_
        num_classes = len(classes_)
        #pass number of features and classes to the model construction 
        #function that will be called when the fit method is called
        self.model_kwargs['num_features'] = num_features
        self.model_kwargs['num_classes'] = num_classes
        #now actually call the optimizer fit method
        TF_Optimizer.fit(self, 
                         train_data=train_data, 
                         train_labels=binarized_labels)
  
    def decision_function(self, test_data):
        #returns what are effectively the margins (for a linear classifier)
        return TF_Optimizer.predict(self, test_data)
    
    def predict(self, test_data):
        #use the one-vs-all rule for multiclass prediction. 
        preds = self.decision_function(test_data)
        preds = np.argmax(preds, axis=1)
        classes_ = self.classes_
        return classes_[preds]

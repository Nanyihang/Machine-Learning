import numpy as np


class TwoLayersNN (object):
    """" TwoLayersNN classifier """

    def __init__ (self, inputDim, hiddenDim, outputDim):
        self.params = dict()
        self.params['w1'] = None
        self.params['b1'] = None
        self.params['w2'] = None
        self.params['b2'] = None
        #########################################################################
        # TODO: 20 points                                                       #
        # - Generate a random NN weight matrix to use to compute loss.          #
        # - By using dictionary (self.params) to store value                    #
        #   with standard normal distribution and Standard deviation = 0.0001.  #
        #########################################################################
        self.params['w1'] = np.random.normal(0.0001, 0.0001, (inputDim, hiddenDim))
        self.params['b1'] = np.ones(hiddenDim)
        self.params['w2'] = np.random.normal(0.0001, 0.0001, (hiddenDim, outputDim))
        self.params['b2'] = np.ones(outputDim)


        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        TwoLayersNN loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to each parameter (w1, b1, w2, b2)
        """
        loss = 0.0
        grads = dict()
        grads['w1'] = None
        grads['b1'] = None
        grads['w2'] = None
        grads['b2'] = None
        #############################################################################
        # TODO: 40 points                                                           #
        # - Compute the NN loss and store to loss variable.                         #
        # - Compute gradient for each parameter and store to grads variable.        #
        # - Use Leaky RELU Activation at hidden and output neurons                  #
        # - Use Softmax loss
        # Note:                                                                     #
        # - Use L2 regularization                                                   #
        # Hint:                                                                     #
        # - Do forward pass and calculate loss value                                #
        # - Do backward pass and calculate derivatives for each weight and bias     #
        #############################################################################
        pass
        w1 = self.params['w1']
        b1 = self.params['b1']
        w2 = self.params['w2']
        b2 = self.params['b2']
        h_prime = x.dot(w1) + b1
        h = np.maximum(0.01 * h_prime, h_prime)
        s_prime = h.dot(w2) + b2
        s = np.maximum(0.01 * s_prime , s_prime)
        exp_s = np.exp(s)
        sum_exp_s = np.sum(exp_s, axis = 1, keepdims = True)
        p_i = exp_s / sum_exp_s
        loss_i = -1 * np.log(p_i[np.arange(x.shape[0]), y])
        hing_loss = np.sum(loss_i) / x.shape[0]
        regularization_loss = reg * np.sum(w1 * w1) + reg * np.sum(w2 * w2)
        loss = hing_loss + regularization_loss
        ds = p_i
        ds[np.arange(x.shape[0]), y] = ds[np.arange(x.shape[0]), y] - 1
        ds = ds / x.shape[0]
        ds_prime = ds.copy()
        ds_prime[[s < 0]] = .01
        ds_prime[[s >= 0]] = 1
        ds_prime = ds_prime * ds
        dw2 = np.dot(h.T, ds_prime)
        dw2 =  dw2 + (2 * reg * w2)
        grads['w2'] = dw2
        grads['b2'] = np.sum(ds_prime, axis = 0)
        dh = np.dot(ds_prime, w2.T)
        dh_prime = dh.copy()
        dh_prime [h < 0] = .01
        dh_prime [h >= 0] = 1
        dh_prime = dh_prime * dh
        dw1 = np.dot(x.T, dh_prime)
        # dw1 = dw1 / x.shape[0]
        dw1 = dw1 + (2 * reg * w1)
        grads['w1'] = dw1
        grads['b1'] = np.sum(dh_prime, axis = 0)
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, grads

    def train (self, x, y, lr=5e-3, reg=5e-3, iterations=100, batchSize=200, decay=0.95, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iterations):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (batchSize, D)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################
            random_index_vector = np.random.choice(x.shape[0], batchSize)  # , replace=True)
            xBatch = x[random_index_vector]
            yBatch = y[random_index_vector]

            # find gradient decent and loss
            loss, gradients = self.calLoss(xBatch, yBatch, reg)
            lossHistory.append(loss)

            self.params['w1'] -= lr * gradients['w1']
            self.params['w2'] -= lr * gradients['w2']
            self.params['b1'] -= lr * gradients['b1']
            self.params['b2'] -= lr * gradients['b2']


            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # Decay learning rate
            lr *= decay
            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Store the predict output in yPred                                    #
        ###########################################################################
        h = x.dot(self.params['w1']) + self.params['b1']
        h = np.maximum(h, .01 * h)
        s = h.dot(self.params['w2']) + self.params['b2']
        s = np.maximum(s, .01 * s)
        yPred = np.argmax(s, axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################


        acc = np.sum(self.predict(x) == y) / x.shape[0]


        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc




import numpy as np
import math


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    prediction = predictions.copy()
    new_exp = np.vectorize(lambda x: math.exp(x))
    if len(predictions.shape)==1:
        pred = prediction-np.max(prediction)
        exp_prob = new_exp(pred)
        probs = np.array(list(map(lambda x: x/np.sum(exp_prob),exp_prob)))
    else:
        pred = list(map(lambda x: x-np.max(x), prediction))
        exp_prob = new_exp(pred)
        probs = np.array(list(map(lambda x: x/np.sum(x),exp_prob)))
    
    return probs

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    new_loss = np.vectorize(lambda x: -math.log(x))
    if len(probs.shape)==1:
        probs_target = probs[target_index]
        size_target = 1
    else:
        batch_size = np.arange(target_index.shape[0])
        probs_target = probs[batch_size,target_index.flatten()]
        size_target = target_index.shape[0]
    loss = np.sum(new_loss(probs_target))/size_target
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    prediction = predictions.copy()
    probs = softmax(prediction) 
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    if len(predictions.shape)==1:
        dprediction[target_index] -= 1
    else:
        batch_size = np.arange(target_index.shape[0])
        dprediction[batch_size,target_index.flatten()] -= 1
        dprediction = dprediction/target_index.shape[0]

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    #loss = reg_strength*np.sum(np.dot(np.transpose(W),W))
    #batch_size = np.arange(W.shape[1])
    #grad = np.array((list(map(lambda x: np.sum(W,axis=1), batch_size))))
    #grad = 2*reg_strength*np.transpose(grad)
    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W

    return loss, grad


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        result = X.copy()
        result[result<0] = 0
        return result

    def backward(self, d_out):
        """
        Backward pass
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out
        d_result[self.X<0] = 0
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        result = np.dot(X,self.W.value)+self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        d_input = np.dot(d_out, np.transpose(self.W.value))
        self.W.grad += np.dot(np.transpose(self.X),d_out)
        self.B.grad += np.sum(d_out, axis=0)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
    
    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X

        out_height = width-self.filter_size+1+2*self.padding
        out_width = width-self.filter_size+1+2*self.padding
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        if self.padding>0:
            self.X_pad = np.zeros((batch_size, height+2*self.padding, width+2*self.padding,channels))
            self.X_pad[:,self.padding:height+self.padding,self.padding:width+self.padding,:] = self.X
        else: self.X_pad = self.X
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        W_step = self.W.value.reshape(self.filter_size*self.filter_size*channels, self.out_channels)
        B_step = np.zeros((batch_size, self.out_channels))
        B_step = np.array(list(map(lambda x: self.B.value, B_step)))
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                X_step = self.X_pad[:,x:x+self.filter_size,y:y+self.filter_size,:]
                X_step = X_step.reshape(batch_size,self.filter_size*self.filter_size*channels)
                result[:,x,y,:] = np.dot(X_step,W_step)+B_step
                
        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        d_input_pad = np.zeros_like(self.X_pad)
        W_step = self.W.value.reshape(self.filter_size*self.filter_size*channels, out_channels)
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                d_out_step = d_out[:,x,y,:].reshape(batch_size,out_channels)
                X_step = self.X_pad[:,x:x+self.filter_size,y:y+self.filter_size,:]
                res = np.dot(d_out_step,np.transpose(W_step)).reshape(X_step.shape)
                d_input_pad[:,x:x+self.filter_size,y:y+self.filter_size,:]+=res
                X_step = X_step.reshape(batch_size,self.filter_size*self.filter_size*channels)
                self.W.grad += np.dot(np.transpose(X_step),d_out_step).reshape(self.W.grad.shape)
                self.B.grad += np.sum(d_out_step, axis=0)
        if self.padding>0:
            d_input = d_input_pad[:,self.padding:height+self.padding,self.padding:width+self.padding,:]
        else: d_input = d_input_pad
        
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool
        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = (height-self.pool_size)//self.stride + 1
        out_width = (width-self.pool_size)//self.stride + 1
        result = np.zeros((batch_size, out_height, out_width, channels))
        self.d_input = np.zeros_like(X)
        for y in range(out_height):
            for x in range(out_width):
                x_step = x*self.stride
                y_step = y*self.stride
                X_step = X[:,x_step:x_step+self.pool_size,y_step:y_step+self.pool_size,:]
                X_step = X_step.reshape(batch_size*channels, self.pool_size*self.pool_size )
                max_el = np.max(X_step, axis=1)
                result[:,x,y,:] = max_el.reshape(batch_size,channels)
                d_step = np.zeros_like(X_step)
                for i in range(batch_size*channels):
                    d_step[i, np.where(X_step[i,:]==max_el[i])] = 1/np.sum(X_step[i,:]==max_el[i])  
                self.d_input[:,x_step:x_step+self.pool_size,y_step:y_step+self.pool_size,:] = d_step.reshape(batch_size, self.pool_size, self.pool_size, channels)
        
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        self.d_input_new = self.d_input
        for y in range(out_height):
            for x in range(out_width):
                x_step = x*self.stride
                y_step = y*self.stride
                d_out_step = d_out[:,x,y,:].reshape(batch_size*channels)
                d_out_big = np.array(list(map(lambda x: np.ones(self.pool_size*self.pool_size)*d_out_step[x], np.arange(batch_size*channels))))
                d_out_big = d_out_big.reshape(batch_size,self.pool_size,self.pool_size,channels)
                self.d_input_new[:,x_step:x_step+self.pool_size,y_step:y_step+self.pool_size,:]*= d_out_big
        return self.d_input_new

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channels = channels
        result = X.reshape(batch_size, height*width*channels)
        return result

    def backward(self, d_out):
        return d_out.reshape(self.batch_size, self.height, self.width, self.channels)

    def params(self):
        # No params!
        return {}
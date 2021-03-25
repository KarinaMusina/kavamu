import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )

class ConvNet:
    """
    Implements a very simple conv net
    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network
        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.layer1 = ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1)
        self.layer2 = ReLULayer()
        self.layer3 = MaxPoolingLayer(4, 4)
        self.layer4 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.layer5 = ReLULayer()
        self.layer6 = MaxPoolingLayer(4, 4)
        self.layer7 = Flattener()
        self.layer8 = FullyConnectedLayer(input_shape[0]*input_shape[1]*conv2_channels//(16*16), n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        params = self.params()
        
        for param_key in params:
            param = params[param_key]
            param.grad = np.zeros_like(param.grad)
            
        step1 = self.layer1.forward(X)
        step2 = self.layer2.forward(step1)
        step3 = self.layer3.forward(step2)
        step4 = self.layer4.forward(step3)
        step5 = self.layer5.forward(step4)
        step6 = self.layer6.forward(step5)
        step7 = self.layer7.forward(step6)
        step8 = self.layer8.forward(step7)
        
        loss, dL = softmax_with_cross_entropy(step8, y)
        
        dstep8 = self.layer8.backward(dL)
        dstep7 = self.layer7.backward(dstep8)
        dstep6 = self.layer6.backward(dstep7)
        dstep5 = self.layer5.backward(dstep6)
        dstep4 = self.layer4.backward(dstep5)
        dstep3 = self.layer3.backward(dstep4)
        dstep2 = self.layer2.backward(dstep3)
        dstep1 = self.layer1.backward(dstep2)
        
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        step1 = self.layer1.forward(X)
        step2 = self.layer2.forward(step1)
        step3 = self.layer3.forward(step2)
        step4 = self.layer4.forward(step3)
        step5 = self.layer5.forward(step4)
        step6 = self.layer6.forward(step5)
        step7 = self.layer7.forward(step6)
        step8 = self.layer8.forward(step7)
        probs = softmax(step8)
        pred = np.array(list(map(lambda x: x.argsort()[-1], probs)))

        return pred

    def params(self):

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        return {'W1': self.layer1.W, 'B1': self.layer1.B, 'W2': self.layer4.W, 'B2': self.layer4.B, 'W3': self.layer8.W, 'B3': self.layer8.B }
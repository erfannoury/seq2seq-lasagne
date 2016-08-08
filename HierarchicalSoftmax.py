from __future__ import division
import numpy as np
import theano.tensor as T
import lasagne
from lasagne import init, nonlinearities
from lasagne.layers.base import Layer

__all__ = [
    "HierarchicalSoftmaxLayer"
]


def custom_h_softmax(x, n_outputs, n_classes, n_outputs_per_class,
              W1, b1, W2, b2, target=None):
    """ Two-level hierarchical softmax.

    The architecture is composed of two softmax layers: the first predicts the
    class of the input x while the second predicts the output of the input x in
    the predicted class.
    More explanations can be found in the original paper [1]_.

    If target is specified, it will only compute the outputs of the
    corresponding targets. Otherwise, if target is None, it will compute all
    the outputs.

    The outputs are grouped in the same order as they are initially defined.

    .. versionadded:: 0.7.1

    Parameters
    ----------
    x: tensor of shape (batch_size, number of features)
        the minibatch input of the two-layer hierarchical softmax.
    batch_size: int
        the size of the minibatch input x.
    n_outputs: int
        the number of outputs.
    n_classes: int
        the number of classes of the two-layer hierarchical softmax. It
        corresponds to the number of outputs of the first softmax. See note at
        the end.
    n_outputs_per_class: int
        the number of outputs per class. See note at the end.
    W1: tensor of shape (number of features of the input x, n_classes)
        the weight matrix of the first softmax, which maps the input x to the
        probabilities of the classes.
    b1: tensor of shape (n_classes,)
        the bias vector of the first softmax layer.
    W2: tensor of shape (n_classes, number of features of the input x, n_outputs_per_class)
        the weight matrix of the second softmax, which maps the input x to
        the probabilities of the outputs.
    b2: tensor of shape (n_classes, n_outputs_per_class)
        the bias vector of the second softmax layer.
    target: tensor of shape either (batch_size,) or (batch_size, 1)
        (optional, default None)
        contains the indices of the targets for the minibatch
        input x. For each input, the function computes the output for its
        corresponding target. If target is None, then all the outputs are
        computed for each input.

    Returns
    -------
    output_probs: tensor of shape (batch_size, n_outputs) or (batch_size, 1)
        Output of the two-layer hierarchical softmax for input x. If target is
        not specified (None), then all the outputs are computed and the
        returned tensor has shape (batch_size, n_outputs). Otherwise, when
        target is specified, only the corresponding outputs are computed and
        the returned tensor has thus shape (batch_size, 1).

    Notes
    -----
    The product of n_outputs_per_class and n_classes has to be greater or equal
    to n_outputs. If it is strictly greater, then the irrelevant outputs will
    be ignored.
    n_outputs_per_class and n_classes have to be the same as the corresponding
    dimensions of the tensors of W1, b1, W2 and b2.
    The most computational efficient configuration is when n_outputs_per_class
    and n_classes are equal to the square root of n_outputs.

    References
    ----------
    .. [1] J. Goodman, "Classes for Fast Maximum Entropy Training,"
        ICASSP, 2001, <http://arxiv.org/abs/cs/0108006>`.
    """
    n_batches = x.shape[0]
    # First softmax that computes the probabilities of belonging to each class
    class_probs = T.nnet.softmax(T.dot(x, W1) + b1) # -> (n_batches, n_classes)

    if target is None:  # Computes the probabilites of all the outputs

        # Second softmax that computes the output probabilities
        activations = T.tensordot(x, W2, (1, 1)) + b2 # -> (n_batches, n_classes, n_outputs_per_class)
        output_probs = T.nnet.softmax(
            activations.reshape((n_batches * n_classes, n_outputs_per_class), ndim=2)) # -> (n_batches * n_classes, n_outputs_per_class)
        output_probs = output_probs.reshape((n_batches, n_classes, n_outputs_per_class), ndim=3) # -> (n_batches, n_classes, n_outputs_per_class)
        output_probs = class_probs.dimshuffle(0, 1, 'x') * output_probs # -> (n_batches, n_classes, 1) * (n_batches, n_classes, n_outputs_per_class)
        output_probs = output_probs.reshape((n_batches, n_classes * n_outputs_per_class), ndim=2)
        # output_probs.shape[1] is n_classes * n_outputs_per_class, which might
        # be greater than n_outputs, so we ignore the potential irrelevant
        # outputs with the next line:
        output_probs = output_probs[:, :n_outputs]

    else:  # Computes the probabilities of the outputs specified by the targets
        raise NotImplementedError("This is not yet implemented")
        pass
        target = target.flatten()

        # Classes to which each target belongs
        target_classes = target // n_outputs_per_class

        # Outputs to which belong each target inside a class
        target_outputs_in_class = target % n_outputs_per_class

        # Second softmax that computes the output probabilities
        activations = T.nnet.blocksparse.sparse_block_dot(
            W2.dimshuffle('x', 0, 1, 2), x.dimshuffle(0, 'x', 1),
            tensor.zeros((n_batches, 1), dtype='int32'), b2,
            target_classes.dimshuffle(0, 'x'))

        output_probs = T.nnet.softmax(activations.dimshuffle(0, 2))
        target_class_probs = class_probs[tensor.arange(batch_size),
                                         target_classes]
        output_probs = output_probs[tensor.arange(batch_size),
                                    target_outputs_in_class]
        output_probs = target_class_probs * output_probs

    return output_probs


class HierarchicalSoftmaxLayer(Layer):
    r"""
    lasagne.layers.HierarchicalSoftmaxLayer(incoming, n_outputs,
    n_classes=None, n_outputs_per_class=None,
    W1=lasagne.init.GlorotNormal(), b1=lasagne.init.Constant(0.),
    W2=lasagne.init.GlorotNormal(), b2=lasagne.init.Constant(0.),
    target=None, **kwargs)

    The Hierarchical Softmax layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    n_outputs : int
        The number of outputs of the layer

    n_classes : int or None
        The number of classes of the two-layer hierarchical softmax.
        It corresponds to the number of outputs of the first softmax.
        If None, it will be calculated from n_outputs.

    n_outputs_per_class : int or None
        the number of outputs per class

    W1 : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(n_features, n_classes)``.
        The weight matrix of the first softmax, which maps the input
        ``incoming`` to the probabilities of the classes.
        See :func:`lasagne.utils.create_param` for more information.

    b1 : Theano shared variable, expression, numpy array, callable
        Initial value, expression or initializer for the biases. If set to
        Biases should be a 1D array with shape ``(n_classes,)``.
        See :func:`lasagne.utils.create_param` for more information.

    W2 : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape
        ``(n_classes, n_features, n_outputs_per_class)``.
        The weight matrix of the second softmax, which maps the input
        ``incoming`` to the probabilities of the outputs
        See :func:`lasagne.utils.create_param` for more information.

    b2 : Theano shared variable, expression, numpy array, callable
        Initial value, expression or initializer for the biases. If set to
        Biases should be a 2D array with shape
        ``(n_classes, n_outputs_per_classes)``.
        See :func:`lasagne.utils.create_param` for more information.

    target: Theano shared variable, expression, numpy array, callable or None
        Contains the indices of the targets for the minibatch input
        ``incoming``. For each input, the function computes the output
        for its corresponding target. If target is ``None``, then all
        the outputs are computed for each input. When provided, Target
        should be of shape ``(n_batch, )`` or ``(n_batch, 1)``.


    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes.

    Returns
    -------
    output_probs : Theano tensor
        Output of the two-layer hierarchical softmax for input ``incoming``.
        If target is not specified (``None``), then all the outputs are computed
        and the returned tensor has shape ``(batch_size, n_outputs)``. Otherwise,
        when target is specified, only the corresponding outputs are computed
        and the returned tensor has thus shape ``(batch_size, 1)``.
    """
    def __init__(self, incoming, n_outputs,
                 n_classes=None, n_outputs_per_class=None,
                 W1=lasagne.init.GlorotNormal(), b1=lasagne.init.Constant(0.),
                 W2=lasagne.init.GlorotNormal(), b2=lasagne.init.Constant(0.),
                 target=None, **kwargs):
        super(HierarchicalSoftmaxLayer, self).__init__(incoming, **kwargs)

        self.n_outputs = n_outputs

        n_features = self.input_shape[1]

        if n_classes is None and n_outputs_per_class is None:
            self.n_classes = int(np.ceil(np.sqrt(self.n_outputs)))
            self.n_outputs_per_class = self.n_classes
        elif n_classes is None:
            self.n_outputs_per_class = n_outputs_per_class
            self.n_classes = int(np.ceil(self.n_outputs * 1.0 / n_outputs_per_class))
        elif n_outputs_per_class is None:
            self.n_classes = n_classes
            self.n_outputs_per_class = int(np.ceil(self.n_outputs * 1.0 / n_classes))
        else:
            self.n_classes = n_classes
            self.n_outputs_per_class = n_outputs_per_class
        if self.n_outputs_per_class * self.n_classes < self.n_outputs:
            raise ValueError("Product of n_outputs_per_class and"
                             "n_classes should not be less than"
                             "n_outputs.")

        self.W1 = self.add_param(W1, (n_features, self.n_classes), name="W1")
        self.W2 = self.add_param(W2, (self.n_classes, n_features,
                                 self.n_outputs_per_class), name="W2")
        self.b1 = self.add_param(b1, (self.n_classes, ), name="b1", regularizable=False)
        self.b2 = self.add_param(b2, (self.n_classes, self.n_outputs_per_class),
                                 name="b2", regularizable=False)

        self.target = target

    def get_output_shape_for(self, input_shape):
        if self.target is None:
            return (input_shape[0], self.n_outputs)
        else:
            return (input_shape[0], 1)


    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            raise ValueError("input dimensionality should be 2")

        output_probs = custom_h_softmax(input,
                                        self.n_outputs, self.n_classes,
                                        self.n_outputs_per_class, self.W1,
                                        self.b1, self.W2, self.b2,
                                        self.target)
        return output_probs
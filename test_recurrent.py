import pytest
from lasagne.layers import InputLayer, DenseLayer, Gate, Layer, EmbeddingLayer
from lasagne.layers import helper
import theano
import theano.tensor as T
import numpy as np
import lasagne
from mock import Mock

from CustomLSTMLayer import LNLSTMLayer, LSTMLayer


def test_lnlstm_get_simple_output():
    feat_size = 20
    hid_size = 10
    l_in = InputLayer((None, None, feat_size))
    l_lstm = LNLSTMLayer(l_in, hid_size)

    output = lasagne.layers.get_output(l_lstm)


def test_lnlstm_get_emb_output():
    hid_size = 10
    inp_size = 10
    out_size = 40
    n_batches = 23
    seqlen = 47
    l_in = InputLayer((n_batches, seqlen), input_var=T.imatrix('input_var'), name="l_in")
    l_emb = EmbeddingLayer(l_in, inp_size, out_size, name="l_emb")
    l_lstm = LNLSTMLayer(l_emb, hid_size, name="l_lstm")

    emb_output = lasagne.layers.get_output(l_emb)
    output = lasagne.layers.get_output(l_lstm)
    output_for = l_lstm.get_output_for([emb_output])


def test_lnlstm_return_shape():
    num_batch, seq_len, n_features1, n_features2 = 5, 3, 10, 11
    num_units = 6
    x = T.tensor4()
    in_shp = (num_batch, seq_len, n_features1, n_features2)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')

    l_lstm = LNLSTMLayer(l_inp, num_units=num_units)
    output = helper.get_output(l_lstm, x)
    output_val = output.eval({x: x_in})
    assert helper.get_output_shape(l_lstm, x_in.shape) == output_val.shape
    assert output_val.shape == (num_batch, seq_len, num_units)


def test_lnlstm_grad():
    num_batch, seq_len, n_features = 5, 3, 10
    num_units = 6
    l_inp = InputLayer((num_batch, seq_len, n_features))
    l_lstm = LNLSTMLayer(l_inp, num_units=num_units)
    output = helper.get_output(l_lstm)
    g = T.grad(T.mean(output), lasagne.layers.get_all_params(l_lstm))
    assert isinstance(g, (list, tuple))


def test_lnlstm_nparams_no_peepholes():
    l_inp = InputLayer((2, 2, 3))
    l_lstm = LNLSTMLayer(l_inp, 5, peepholes=False, learn_init=False)

    # 7*n_gates
    # the 7 is because we have  hid_to_gate, in_to_gate and bias, and alpha 
    # and beta for hid_to_gate and in_to_gate for LN, for each gate
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 30

    # bias params + two betas + init params
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 15


def test_lnlstm_nparams_peepholes():
    l_inp = InputLayer((2, 2, 3))
    l_lstm = LNLSTMLayer(l_inp, 5, peepholes=True, learn_init=False)

    # 7*n_gates + peepholes(3*(1+2)).
    # the 7 is because we have  hid_to_gate, in_to_gate and bias and 
    # layer normalization for each gate
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 33

    # bias params(3*#gate) + peephole(3) + init params(2)
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 15


def test_lnlstm_nparams_learn_init():
    l_inp = InputLayer((2, 2, 3))
    l_lstm = LNLSTMLayer(l_inp, 5, peepholes=False, learn_init=True)

    # 7*n_gates + inits(2).
    # the 7 is because we have  hid_to_gate, in_to_gate and bias and
    # layer normalization for each gate
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 32

    # bias params(3*#gate) + init params(2)
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 15


def test_lnlstm_hid_init_layer():
    # test that you can set hid_init to be a layer
    l_inp = InputLayer((2, 2, 3))
    l_inp_h = InputLayer((2, 5))
    l_cell_h = InputLayer((2, 5))
    l_lstm = LNLSTMLayer(l_inp, 5, hid_init=l_inp_h, cell_init=l_cell_h)

    x = T.tensor3()
    h = T.matrix()

    output = lasagne.layers.get_output(l_lstm, {l_inp: x, l_inp_h: h})


def test_lnlstm_nparams_hid_init_layer():
    # test that you can see layers through hid_init
    l_inp = InputLayer((2, 2, 3))
    l_inp_h = InputLayer((2, 5))
    l_inp_h_de = DenseLayer(l_inp_h, 7)
    l_inp_cell = InputLayer((2, 5))
    l_inp_cell_de = DenseLayer(l_inp_cell, 7)
    l_lstm = LNLSTMLayer(l_inp, 7, hid_init=l_inp_h_de, cell_init=l_inp_cell_de)

    # directly check the layers can be seen through hid_init
    layers_to_find = [l_inp, l_inp_h, l_inp_h_de, l_inp_cell, l_inp_cell_de,
                      l_lstm]
    assert lasagne.layers.get_all_layers(l_lstm) == layers_to_find

    # 7*n_gates + 3*n_peepholes + 4
    # the 7 is because we have  hid_to_gate, in_to_gate and bias and 
    # layer normalization for each gate
    # 4 is for the W and b parameters in the two DenseLayer layers
    print lasagne.layers.get_all_params(l_lstm, trainable=True)
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 37

    # LSTM bias params(4) + LN betas(2*#gate) (+ Dense bias params(1) * 2
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 15


def test_lnlstm_hid_init_mask():
    # test that you can set hid_init to be a layer when a mask is provided
    l_inp = InputLayer((2, 2, 3))
    l_inp_h = InputLayer((2, 5))
    l_inp_msk = InputLayer((2, 2))
    l_cell_h = InputLayer((2, 5))
    l_lstm = LNLSTMLayer(l_inp, 5, hid_init=l_inp_h, mask_input=l_inp_msk,
                       cell_init=l_cell_h)

    x = T.tensor3()
    h = T.matrix()
    msk = T.matrix()

    inputs = {l_inp: x, l_inp_h: h, l_inp_msk: msk}
    output = lasagne.layers.get_output(l_lstm, inputs)


def test_lnlstm_hid_init_layer_eval():
    # Test `hid_init` as a `Layer` with some dummy input. Compare the output of
    # a network with a `Layer` as input to `hid_init` to a network with a
    # `np.array` as input to `hid_init`
    n_units = 7
    n_test_cases = 2
    in_shp = (n_test_cases, 2, 3)
    in_h_shp = (1, n_units)
    in_cell_shp = (1, n_units)

    # dummy inputs
    X_test = np.ones(in_shp, dtype=theano.config.floatX)
    Xh_test = np.ones(in_h_shp, dtype=theano.config.floatX)
    Xc_test = np.ones(in_cell_shp, dtype=theano.config.floatX)
    Xh_test_batch = np.tile(Xh_test, (n_test_cases, 1))
    Xc_test_batch = np.tile(Xc_test, (n_test_cases, 1))

    # network with `Layer` initializer for hid_init
    l_inp = InputLayer(in_shp)
    l_inp_h = InputLayer(in_h_shp)
    l_inp_cell = InputLayer(in_cell_shp)
    l_rec_inp_layer = LNLSTMLayer(l_inp, n_units, hid_init=l_inp_h,
                                cell_init=l_inp_cell, nonlinearity=None)

    # network with `np.array` initializer for hid_init
    l_rec_nparray = LNLSTMLayer(l_inp, n_units, hid_init=Xh_test,
                              cell_init=Xc_test, nonlinearity=None)

    # copy network parameters from l_rec_inp_layer to l_rec_nparray
    l_il_param = dict([(p.name, p) for p in l_rec_inp_layer.get_params()])
    l_rn_param = dict([(p.name, p) for p in l_rec_nparray.get_params()])
    for k, v in l_rn_param.items():
        if k in l_il_param:
            v.set_value(l_il_param[k].get_value())

    # build the theano functions
    X = T.tensor3()
    Xh = T.matrix()
    Xc = T.matrix()
    output_inp_layer = lasagne.layers.get_output(l_rec_inp_layer,
                                                 {l_inp: X, l_inp_h:
                                                  Xh, l_inp_cell: Xc})
    output_nparray = lasagne.layers.get_output(l_rec_nparray, {l_inp: X})

    # test both nets with dummy input
    output_val_inp_layer = output_inp_layer.eval({X: X_test, Xh: Xh_test_batch,
                                                  Xc: Xc_test_batch})
    output_val_nparray = output_nparray.eval({X: X_test})

    # check output given `Layer` is the same as with `np.array`
    assert np.allclose(output_val_inp_layer, output_val_nparray)


def test_lnlstm_grad_clipping():
    # test that you can set grad_clip variable
    x = T.tensor3()
    l_rec = LNLSTMLayer(InputLayer((2, 2, 3)), 5, grad_clipping=1)
    output = lasagne.layers.get_output(l_rec, x)


def test_lnlstm_bck():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.ones(in_shp).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_lstm_fwd = LNLSTMLayer(l_inp, num_units=num_units, backwards=False)
    lasagne.random.get_rng().seed(1234)
    l_lstm_bck = LNLSTMLayer(l_inp, num_units=num_units, backwards=True)
    output_fwd = helper.get_output(l_lstm_fwd, x)
    output_bck = helper.get_output(l_lstm_bck, x)

    output_fwd_val = output_fwd.eval({x: x_in})
    output_bck_val = output_bck.eval({x: x_in})

    # test that the backwards model reverses its final input
    np.testing.assert_almost_equal(output_fwd_val, output_bck_val[:, ::-1])


def test_lnlstm_precompute():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    l_mask_inp = InputLayer(in_shp[:2])

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones((num_batch, seq_len), dtype='float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_lstm_precompute = LNLSTMLayer(
        l_inp, num_units=num_units, precompute_input=True,
        mask_input=l_mask_inp)
    lasagne.random.get_rng().seed(1234)
    l_lstm_no_precompute = LNLSTMLayer(
        l_inp, num_units=num_units, precompute_input=False,
        mask_input=l_mask_inp)
    output_precompute = helper.get_output(
        l_lstm_precompute).eval({l_inp.input_var: x_in,
                                 l_mask_inp.input_var: mask_in})
    output_no_precompute = helper.get_output(
        l_lstm_no_precompute).eval({l_inp.input_var: x_in,
                                    l_mask_inp.input_var: mask_in})

    # test that the backwards model reverses its final input
    np.testing.assert_almost_equal(output_precompute, output_no_precompute)


def test_lnlstm_variable_input_size():
    # that seqlen and batchsize None works
    num_batch, n_features1 = 6, 5
    num_units = 13
    x = T.tensor3()

    in_shp = (None, None, n_features1)
    l_inp = InputLayer(in_shp)
    x_in1 = np.ones((num_batch+1, 3+1, n_features1)).astype('float32')
    x_in2 = np.ones((num_batch, 3, n_features1)).astype('float32')
    l_rec = LNLSTMLayer(l_inp, num_units=num_units, backwards=False)
    output = helper.get_output(l_rec, x)
    output_val1 = output.eval({x: x_in1})
    output_val2 = output.eval({x: x_in2})


def test_lnlstm_unroll_scan_fwd():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    l_mask_inp = InputLayer(in_shp[:2])

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones(in_shp[:2]).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_lstm_scan = LNLSTMLayer(l_inp, num_units=num_units, backwards=False,
                            unroll_scan=False, mask_input=l_mask_inp)
    lasagne.random.get_rng().seed(1234)
    l_lstm_unrolled = LNLSTMLayer(l_inp, num_units=num_units, backwards=False,
                                unroll_scan=True, mask_input=l_mask_inp)
    output_scan = helper.get_output(l_lstm_scan)
    output_unrolled = helper.get_output(l_lstm_unrolled)

    output_scan_val = output_scan.eval({l_inp.input_var: x_in,
                                        l_mask_inp.input_var: mask_in})
    output_unrolled_val = output_unrolled.eval({l_inp.input_var: x_in,
                                                l_mask_inp.input_var: mask_in})

    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_lnlstm_unroll_scan_bck():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_lstm_scan = LNLSTMLayer(l_inp, num_units=num_units, backwards=True,
                            unroll_scan=False)
    lasagne.random.get_rng().seed(1234)
    l_lstm_unrolled = LNLSTMLayer(l_inp, num_units=num_units, backwards=True,
                                unroll_scan=True)
    output_scan = helper.get_output(l_lstm_scan, x)
    output_scan_unrolled = helper.get_output(l_lstm_unrolled, x)

    output_scan_val = output_scan.eval({x: x_in})
    output_unrolled_val = output_scan_unrolled.eval({x: x_in})

    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_lnlstm_passthrough():
    # Tests that the LSTM can simply pass through its input
    l_in = InputLayer((4, 5, 6))
    zero = lasagne.init.Constant(0.)
    one = lasagne.init.Constant(1.)
    pass_gate = Gate(zero, zero, zero, one, None)
    no_gate = Gate(zero, zero, zero, zero, None)
    in_pass_gate = Gate(
        np.eye(6).astype(theano.config.floatX), zero, zero, zero, None)
    l_rec = LNLSTMLayer(
        l_in, 6, pass_gate, no_gate, in_pass_gate, pass_gate, None)
    out = lasagne.layers.get_output(l_rec)
    inp = np.arange(4*5*6).reshape(4, 5, 6).astype(theano.config.floatX)
    # np.testing.assert_almost_equal(out.eval({l_in.input_var: inp}), inp)


def test_lnlstm_return_final():
    num_batch, seq_len, n_features = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features)
    x_in = np.random.random(in_shp).astype('float32')

    l_inp = InputLayer(in_shp)
    lasagne.random.get_rng().seed(1234)
    l_rec_final = LNLSTMLayer(l_inp, num_units, only_return_final=True)
    lasagne.random.get_rng().seed(1234)
    l_rec_all = LNLSTMLayer(l_inp, num_units, only_return_final=False)

    output_final = helper.get_output(l_rec_final).eval({l_inp.input_var: x_in})
    output_all = helper.get_output(l_rec_all).eval({l_inp.input_var: x_in})

    assert output_final.shape == (output_all.shape[0], output_all.shape[2])
    assert output_final.shape == lasagne.layers.get_output_shape(l_rec_final)
    assert np.allclose(output_final, output_all[:, -1])


def test_lstm_get_simple_output():
    feat_size = 20
    hid_size = 10
    l_in = InputLayer((None, None, feat_size))
    l_lstm = LSTMLayer(l_in, hid_size)

    output = lasagne.layers.get_output(l_lstm)


def test_lstm_get_emb_output():
    hid_size = 10
    inp_size = 10
    out_size = 40
    l_in = InputLayer((None, None), input_var=T.imatrix())
    l_emb = EmbeddingLayer(l_in, inp_size, out_size)
    l_lstm = LSTMLayer(l_emb, hid_size)

    emb_output = lasagne.layers.get_output(l_emb)
    output = lasagne.layers.get_output(l_lstm)
    output_for = l_lstm.get_output_for([emb_output])


def test_lstm_return_shape():
    num_batch, seq_len, n_features1, n_features2 = 5, 3, 10, 11
    num_units = 6
    x = T.tensor4()
    in_shp = (num_batch, seq_len, n_features1, n_features2)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')

    l_lstm = LSTMLayer(l_inp, num_units=num_units)
    output = helper.get_output(l_lstm, x)
    output_val = output.eval({x: x_in})
    assert helper.get_output_shape(l_lstm, x_in.shape) == output_val.shape
    assert output_val.shape == (num_batch, seq_len, num_units)


def test_lstm_grad():
    num_batch, seq_len, n_features = 5, 3, 10
    num_units = 6
    l_inp = InputLayer((num_batch, seq_len, n_features))
    l_lstm = LSTMLayer(l_inp, num_units=num_units)
    output = helper.get_output(l_lstm)
    g = T.grad(T.mean(output), lasagne.layers.get_all_params(l_lstm))
    assert isinstance(g, (list, tuple))


def test_lstm_nparams_no_peepholes():
    l_inp = InputLayer((2, 2, 3))
    l_lstm = LSTMLayer(l_inp, 5, peepholes=False, learn_init=False)

    # 3*n_gates
    # the 3 is because we have  hid_to_gate, in_to_gate and bias for each gate
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 12

    # bias params + init params
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 6


def test_lstm_nparams_peepholes():
    l_inp = InputLayer((2, 2, 3))
    l_lstm = LSTMLayer(l_inp, 5, peepholes=True, learn_init=False)

    # 3*n_gates + peepholes(3).
    # the 3 is because we have  hid_to_gate, in_to_gate and bias for each gate
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 15

    # bias params(4) + init params(2)
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 6


def test_lstm_nparams_learn_init():
    l_inp = InputLayer((2, 2, 3))
    l_lstm = LSTMLayer(l_inp, 5, peepholes=False, learn_init=True)

    # 3*n_gates + inits(2).
    # the 3 is because we have  hid_to_gate, in_to_gate and bias for each gate
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 14

    # bias params(4) + init params(2)
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 6


def test_lstm_hid_init_layer():
    # test that you can set hid_init to be a layer
    l_inp = InputLayer((2, 2, 3))
    l_inp_h = InputLayer((2, 5))
    l_cell_h = InputLayer((2, 5))
    l_lstm = LSTMLayer(l_inp, 5, hid_init=l_inp_h, cell_init=l_cell_h)

    x = T.tensor3()
    h = T.matrix()

    output = lasagne.layers.get_output(l_lstm, {l_inp: x, l_inp_h: h})


def test_lstm_nparams_hid_init_layer():
    # test that you can see layers through hid_init
    l_inp = InputLayer((2, 2, 3))
    l_inp_h = InputLayer((2, 5))
    l_inp_h_de = DenseLayer(l_inp_h, 7)
    l_inp_cell = InputLayer((2, 5))
    l_inp_cell_de = DenseLayer(l_inp_cell, 7)
    l_lstm = LSTMLayer(l_inp, 7, hid_init=l_inp_h_de, cell_init=l_inp_cell_de)

    # directly check the layers can be seen through hid_init
    layers_to_find = [l_inp, l_inp_h, l_inp_h_de, l_inp_cell, l_inp_cell_de,
                      l_lstm]
    assert lasagne.layers.get_all_layers(l_lstm) == layers_to_find

    # 3*n_gates + 4
    # the 3 is because we have  hid_to_gate, in_to_gate and bias for each gate
    # 4 is for the W and b parameters in the two DenseLayer layers
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 19

    # GRU bias params(3) + Dense bias params(1) * 2
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 6


def test_lstm_hid_init_mask():
    # test that you can set hid_init to be a layer when a mask is provided
    l_inp = InputLayer((2, 2, 3))
    l_inp_h = InputLayer((2, 5))
    l_inp_msk = InputLayer((2, 2))
    l_cell_h = InputLayer((2, 5))
    l_lstm = LSTMLayer(l_inp, 5, hid_init=l_inp_h, mask_input=l_inp_msk,
                       cell_init=l_cell_h)

    x = T.tensor3()
    h = T.matrix()
    msk = T.matrix()

    inputs = {l_inp: x, l_inp_h: h, l_inp_msk: msk}
    output = lasagne.layers.get_output(l_lstm, inputs)


def test_lstm_hid_init_layer_eval():
    # Test `hid_init` as a `Layer` with some dummy input. Compare the output of
    # a network with a `Layer` as input to `hid_init` to a network with a
    # `np.array` as input to `hid_init`
    n_units = 7
    n_test_cases = 2
    in_shp = (n_test_cases, 2, 3)
    in_h_shp = (1, n_units)
    in_cell_shp = (1, n_units)

    # dummy inputs
    X_test = np.ones(in_shp, dtype=theano.config.floatX)
    Xh_test = np.ones(in_h_shp, dtype=theano.config.floatX)
    Xc_test = np.ones(in_cell_shp, dtype=theano.config.floatX)
    Xh_test_batch = np.tile(Xh_test, (n_test_cases, 1))
    Xc_test_batch = np.tile(Xc_test, (n_test_cases, 1))

    # network with `Layer` initializer for hid_init
    l_inp = InputLayer(in_shp)
    l_inp_h = InputLayer(in_h_shp)
    l_inp_cell = InputLayer(in_cell_shp)
    l_rec_inp_layer = LSTMLayer(l_inp, n_units, hid_init=l_inp_h,
                                cell_init=l_inp_cell, nonlinearity=None)

    # network with `np.array` initializer for hid_init
    l_rec_nparray = LSTMLayer(l_inp, n_units, hid_init=Xh_test,
                              cell_init=Xc_test, nonlinearity=None)

    # copy network parameters from l_rec_inp_layer to l_rec_nparray
    l_il_param = dict([(p.name, p) for p in l_rec_inp_layer.get_params()])
    l_rn_param = dict([(p.name, p) for p in l_rec_nparray.get_params()])
    for k, v in l_rn_param.items():
        if k in l_il_param:
            v.set_value(l_il_param[k].get_value())

    # build the theano functions
    X = T.tensor3()
    Xh = T.matrix()
    Xc = T.matrix()
    output_inp_layer = lasagne.layers.get_output(l_rec_inp_layer,
                                                 {l_inp: X, l_inp_h:
                                                  Xh, l_inp_cell: Xc})
    output_nparray = lasagne.layers.get_output(l_rec_nparray, {l_inp: X})

    # test both nets with dummy input
    output_val_inp_layer = output_inp_layer.eval({X: X_test, Xh: Xh_test_batch,
                                                  Xc: Xc_test_batch})
    output_val_nparray = output_nparray.eval({X: X_test})

    # check output given `Layer` is the same as with `np.array`
    assert np.allclose(output_val_inp_layer, output_val_nparray)


def test_lstm_grad_clipping():
    # test that you can set grad_clip variable
    x = T.tensor3()
    l_rec = LSTMLayer(InputLayer((2, 2, 3)), 5, grad_clipping=1)
    output = lasagne.layers.get_output(l_rec, x)


def test_lstm_bck():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.ones(in_shp).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_lstm_fwd = LSTMLayer(l_inp, num_units=num_units, backwards=False)
    lasagne.random.get_rng().seed(1234)
    l_lstm_bck = LSTMLayer(l_inp, num_units=num_units, backwards=True)
    output_fwd = helper.get_output(l_lstm_fwd, x)
    output_bck = helper.get_output(l_lstm_bck, x)

    output_fwd_val = output_fwd.eval({x: x_in})
    output_bck_val = output_bck.eval({x: x_in})

    # test that the backwards model reverses its final input
    np.testing.assert_almost_equal(output_fwd_val, output_bck_val[:, ::-1])


def test_lstm_precompute():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    l_mask_inp = InputLayer(in_shp[:2])

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones((num_batch, seq_len), dtype='float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_lstm_precompute = LSTMLayer(
        l_inp, num_units=num_units, precompute_input=True,
        mask_input=l_mask_inp)
    lasagne.random.get_rng().seed(1234)
    l_lstm_no_precompute = LSTMLayer(
        l_inp, num_units=num_units, precompute_input=False,
        mask_input=l_mask_inp)
    output_precompute = helper.get_output(
        l_lstm_precompute).eval({l_inp.input_var: x_in,
                                 l_mask_inp.input_var: mask_in})
    output_no_precompute = helper.get_output(
        l_lstm_no_precompute).eval({l_inp.input_var: x_in,
                                    l_mask_inp.input_var: mask_in})

    # test that the backwards model reverses its final input
    np.testing.assert_almost_equal(output_precompute, output_no_precompute)


def test_lstm_variable_input_size():
    # that seqlen and batchsize None works
    num_batch, n_features1 = 6, 5
    num_units = 13
    x = T.tensor3()

    in_shp = (None, None, n_features1)
    l_inp = InputLayer(in_shp)
    x_in1 = np.ones((num_batch+1, 3+1, n_features1)).astype('float32')
    x_in2 = np.ones((num_batch, 3, n_features1)).astype('float32')
    l_rec = LSTMLayer(l_inp, num_units=num_units, backwards=False)
    output = helper.get_output(l_rec, x)
    output_val1 = output.eval({x: x_in1})
    output_val2 = output.eval({x: x_in2})


def test_lstm_unroll_scan_fwd():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    l_mask_inp = InputLayer(in_shp[:2])

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones(in_shp[:2]).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_lstm_scan = LSTMLayer(l_inp, num_units=num_units, backwards=False,
                            unroll_scan=False, mask_input=l_mask_inp)
    lasagne.random.get_rng().seed(1234)
    l_lstm_unrolled = LSTMLayer(l_inp, num_units=num_units, backwards=False,
                                unroll_scan=True, mask_input=l_mask_inp)
    output_scan = helper.get_output(l_lstm_scan)
    output_unrolled = helper.get_output(l_lstm_unrolled)

    output_scan_val = output_scan.eval({l_inp.input_var: x_in,
                                        l_mask_inp.input_var: mask_in})
    output_unrolled_val = output_unrolled.eval({l_inp.input_var: x_in,
                                                l_mask_inp.input_var: mask_in})

    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_lstm_unroll_scan_bck():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_lstm_scan = LSTMLayer(l_inp, num_units=num_units, backwards=True,
                            unroll_scan=False)
    lasagne.random.get_rng().seed(1234)
    l_lstm_unrolled = LSTMLayer(l_inp, num_units=num_units, backwards=True,
                                unroll_scan=True)
    output_scan = helper.get_output(l_lstm_scan, x)
    output_scan_unrolled = helper.get_output(l_lstm_unrolled, x)

    output_scan_val = output_scan.eval({x: x_in})
    output_unrolled_val = output_scan_unrolled.eval({x: x_in})

    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_lstm_passthrough():
    # Tests that the LSTM can simply pass through its input
    l_in = InputLayer((4, 5, 6))
    zero = lasagne.init.Constant(0.)
    one = lasagne.init.Constant(1.)
    pass_gate = Gate(zero, zero, zero, one, None)
    no_gate = Gate(zero, zero, zero, zero, None)
    in_pass_gate = Gate(
        np.eye(6).astype(theano.config.floatX), zero, zero, zero, None)
    l_rec = LSTMLayer(
        l_in, 6, pass_gate, no_gate, in_pass_gate, pass_gate, None)
    out = lasagne.layers.get_output(l_rec)
    inp = np.arange(4*5*6).reshape(4, 5, 6).astype(theano.config.floatX)
    np.testing.assert_almost_equal(out.eval({l_in.input_var: inp}), inp)


def test_lstm_return_final():
    num_batch, seq_len, n_features = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features)
    x_in = np.random.random(in_shp).astype('float32')

    l_inp = InputLayer(in_shp)
    lasagne.random.get_rng().seed(1234)
    l_rec_final = LSTMLayer(l_inp, num_units, only_return_final=True)
    lasagne.random.get_rng().seed(1234)
    l_rec_all = LSTMLayer(l_inp, num_units, only_return_final=False)

    output_final = helper.get_output(l_rec_final).eval({l_inp.input_var: x_in})
    output_all = helper.get_output(l_rec_all).eval({l_inp.input_var: x_in})

    assert output_final.shape == (output_all.shape[0], output_all.shape[2])
    assert output_final.shape == lasagne.layers.get_output_shape(l_rec_final)
    assert np.allclose(output_final, output_all[:, -1])

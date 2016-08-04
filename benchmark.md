I ran the simple recurrent example in the Lasagne's examples folder using the `RecurrentLayer`, `LSTMLayer`, and `LNLSTMLayer`. Although `LNLSTMLayer` consistently is better than `LSTMLayer`, however, they both perform worse than the `RecurrentLayer`. I don't know why (is it the number of parameters?!)

## `LNLSTMLayer`
```
Epoch 0 validation cost = 0.164214223623
Epoch 1 validation cost = 0.159697681665
Epoch 2 validation cost = 0.158522546291
Epoch 3 validation cost = 0.149368345737
Epoch 4 validation cost = 0.143195211887
Epoch 5 validation cost = 0.133679702878
Epoch 6 validation cost = 0.116920441389
Epoch 7 validation cost = 0.115493454039
Epoch 8 validation cost = 0.107204169035
Epoch 9 validation cost = 0.0983866751194
```

## `LSTMLayer`
```
Epoch 0 validation cost = 0.172208026052
Epoch 1 validation cost = 0.167089700699
Epoch 2 validation cost = 0.164001345634
Epoch 3 validation cost = 0.160704404116
Epoch 4 validation cost = 0.158061087132
Epoch 5 validation cost = 0.156837075949
Epoch 6 validation cost = 0.155003786087
Epoch 7 validation cost = 0.155499473214
Epoch 8 validation cost = 0.154736429453
Epoch 9 validation cost = 0.153016999364
```

## `RecurrentLayer`
```
Epoch 0 validation cost = 0.105915859342
Epoch 1 validation cost = 0.0711021050811
Epoch 2 validation cost = 0.0434408932924
Epoch 3 validation cost = 0.0317240469158
Epoch 4 validation cost = 0.0254634469748
Epoch 5 validation cost = 0.0166137181222
Epoch 6 validation cost = 0.0152615336701
Epoch 7 validation cost = 0.0146559709683
Epoch 8 validation cost = 0.0126030342653
Epoch 9 validation cost = 0.0150570366532
```

**Note**: I used the default parameters for `LNLSTMLayer` and `LSTMLayer`.
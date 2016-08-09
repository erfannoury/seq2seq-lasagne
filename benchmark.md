I ran the simple recurrent example in the Lasagne's examples folder using the `RecurrentLayer`, `LSTMLayer`, and `LNLSTMLayer`. As can be seen, `LNLSTMLayer` with default parameters is on par with the vanilla `RecurrentLayer`, however, by increasing its learning rate, `LNLSTMLayer` yields better results than both `LSTMLayer` and `RecurrentLayer`.

## `LNLSTMLayer`
`learning rate = 0.001`
```
Epoch 0 validation cost = 0.124967135489
Epoch 1 validation cost = 0.104221150279
Epoch 2 validation cost = 0.113805517554
Epoch 3 validation cost = 0.0695573017001
Epoch 4 validation cost = 0.0419696755707
Epoch 5 validation cost = 0.034377027303
Epoch 6 validation cost = 0.0362087339163
Epoch 7 validation cost = 0.0239392705262
Epoch 8 validation cost = 0.0198697652668
Epoch 9 validation cost = 0.0207334179431
```
`learning rate = 0.01`
```
Epoch 0 validation cost = 0.0719093978405
Epoch 1 validation cost = 0.0238488707691
Epoch 2 validation cost = 0.0304331239313
Epoch 3 validation cost = 0.0157626084983
Epoch 4 validation cost = 0.0185124706477
Epoch 5 validation cost = 0.0174685548991
Epoch 6 validation cost = 0.00522942654788
Epoch 7 validation cost = 0.0170272365212
Epoch 8 validation cost = 0.0143625522032
Epoch 9 validation cost = 0.00972528476268
```

## `LSTMLayer`
`learning rate = 0.001`
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
`learning rate = 0.01`
```
Epoch 0 validation cost = 0.149486139417
Epoch 1 validation cost = 0.132910132408
Epoch 2 validation cost = 0.110643230379
Epoch 3 validation cost = 0.082837626338
Epoch 4 validation cost = 0.0543539375067
Epoch 5 validation cost = 0.0605337098241
Epoch 6 validation cost = 0.0141272759065
Epoch 7 validation cost = 0.0137599119917
Epoch 8 validation cost = 0.0134349633008
Epoch 9 validation cost = 0.00794639345258
```

## `RecurrentLayer`
`learning rate = 0.001`
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
`learning rate = 0.01`
```
Epoch 0 validation cost = 0.0928676426411
Epoch 1 validation cost = 0.0890067070723
Epoch 2 validation cost = 0.0798841491342
Epoch 3 validation cost = 0.0823505669832
Epoch 4 validation cost = 0.0796484351158
Epoch 5 validation cost = 0.0873657986522
Epoch 6 validation cost = 0.0808195844293
Epoch 7 validation cost = 0.0882367044687
Epoch 8 validation cost = 0.0778567269444
Epoch 9 validation cost = 0.0796619132161
```

**Note**: I used the default parameters for `LNLSTMLayer` and `LSTMLayer`.
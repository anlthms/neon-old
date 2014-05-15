#using cuda-convnet to train on MNIST
#benchmark performance from LeCun 1998 was 0.95% without distortions

#todo: convert mnist data to colmajor
    #pickled data files [done]
    #data provider class and change in command line below
#todo: debug layer and params files

#consider changing #define MAX_DATA_ON_GPU in util.cuh

#Train
python convnet.py --data-path=./MNIST/ --save-path=./MNIST/saved/ --test-range=6 --train-range=1-5 --layer-def=./example-layers/layers-conv-local-mnist.cfg --layer-params=./example-layers/layer-params-conv-local-mnist.cfg --data-provider=mnist --test-freq=13 --crop-border=4 --epochs=10

#Test:
#todo: replace the model
python convnet.py -f ./MNIST/saved/ConvNet__2011-12-17_18.13.52 --multiview-test=1 --test-only=1 --logreg-name=logprob --test-range=7

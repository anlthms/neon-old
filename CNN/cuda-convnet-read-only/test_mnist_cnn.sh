#using cuda-convnet to train on MNIST
#benchmark performance from LeCun 1998 was 0.95% without distortions

#todo: convert mnist data to colmajor
    #pickled data files
    #data provider class and change in command line below
#todo: debug layer and params files

#Train
python convnet.py --data-path=/usr/local/data/datasets/cifar10/cifar-10-py-colmajor/ --save-path=/usr/local/data/datasets/cifar10/tmp --test-range=5 --train-range=1-4 --layer-def=./example-layers/layers-conv-local-mnist.cfg --layer-params=./example-layers/layer-params-conv-local-mnist.cfg --data-provider=cifar-cropped --test-freq=13 --crop-border=4 --epochs=100

#Test:
#todo: replace the model
python convnet.py -f /storage2/tmp/ConvNet__2011-12-17_18.13.52 --multiview-test=1 --test-only=1 --logreg-name=logprob --test-range=6

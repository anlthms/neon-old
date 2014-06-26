#using cuda-convnet to train on MNIST
#benchmark performance from LeCun 1998 was 0.95% without distortions

#convert mnist data to colmajor
    #pickled data files [done]
    #data provider class and change in command line below [done]
#todo: debug layer and params files 

#consider changing #define MAX_DATA_ON_GPU in util.cuh

#Train
/usr/bin/python convnet.py --data-path=./MNIST/ --save-path=./MNIST/saved/ --test-range=6 --train-range=1-6 --layer-def=./example-layers/layers-conv-local-mnist.cfg --layer-params=./example-layers/layer-params-conv-local-mnist.cfg --data-provider=mnist --test-freq=5 --epochs=50

#to resume training , change cfg file, make sure ConvNet file is the latest one
#add the validation set in and do more training epochs
python convnet.py -f ./MNIST/saved/ConvNet__2014-05-15_18.15.46  --train-range=1-6 --epochs=10
#more epochs, with optionaly changing the cfg file learning rates
cp ./MNIST/saved/MNIST/saved/ConvNet__2014-05-15_18.15.46/* ./MNIST/saved/ConvNet__2014-05-15_18.15.46/
python convnet.py -f ./MNIST/saved/ConvNet__2014-05-15_18.15.46  --epochs=60 

#Test:
#todo: replace the model
cp ./MNIST/saved/MNIST/saved/ConvNet__2014-05-15_18.15.46/* ./MNIST/saved/ConvNet__2014-05-15_18.15.46/
python convnet.py -f ./MNIST/saved/ConvNet__2014-05-15_18.15.46 --test-only=1 --logreg-name=logprob --test-range=7

#plotting cost function
python shownet.py -f ./MNIST/saved/ConvNet__2014-05-15_16.15.47 --show-cost=logprob --cost-idx=1

#visualize learned filters
python shownet.py -f ./MNIST/saved/ConvNet__2014-05-15_16.15.47 --show-filters=conv1 --no-rgb=1

#view test case predictions
python shownet.py -f ./MNIST/saved/ConvNet__2014-05-15_16.15.47 --show-preds=probs

#show errors:
python shownet.py -f ./MNIST/saved/ConvNet__2014-05-15_16.15.47 --show-preds=probs --only-errors=1

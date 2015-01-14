# this will be given parameters eps, mom, weightWidth, create a cfg, run a model, read and return the model output.
import os
os.chdir('/home/local/code/cuda-convnet2/')
print os.getcwd()
import sys
import time
sys.path.append('/home/local/code/cuda-convnet2/')
from python_util.gpumodel import IGPUModel
from ipdb import set_trace as trace

def call_convnet(params): # 
 
	params['mom'] = 0.9
	params['nonlins'] = 'relu' # 'logistic'
	#params['epochs'] = (20,)
	params['steps'] = (3,5)

	timestring = str(int(time.time()))

	param_file = 'spearmint/configs/sm-params' + timestring + '.cfg'
	write_params(param_file, params)

	layer_file = 'spearmint/configs/sm-layers' + timestring + '.cfg'
	write_layers(layer_file, params)
	#layer_file = 'layers/layers-cifar10-urs.cfg'

	# Part 2: Run a model
	datapath='/usr/local/data/datasets/cifar-10-py-colmajor' # contains batch 1-6
	savepath='output'

	callstring = " python convnet.py \
	    --data-provider cifar \
	    --test-range 6 \
	    --train-range 1-5 \
	    --epochs "+str(params['epochs'][0])+" \
	    --data-path "+datapath+" \
	    --inner-size 24 \
	    --save-path "+savepath+" \
	    --test-freq 50 \
	    --gpu 0 \
	    --mini 128 \
	    --layer-def "+layer_file+" \
	    --layer-params "+param_file+" \
	    --save-path ./output/ConvNet_spearmint_" + timestring # save-file?!?

	
	os.system(callstring) 


	# Part 3: Read the model output

	modelname =  sorted(os.listdir("./output/ConvNet_spearmint_" + timestring))[-1] # this is a problem: 
	model = IGPUModel.load_checkpoint("./output/ConvNet_spearmint_" + timestring + "/" + modelname)
	print model['model_state'].keys()
	result = model['model_state']['test_outputs'][-1][0]['logprob'][1]
	return result/10000.


# Write a function like this called 'main'
def main(job_id, params):
	print 'Anything printed here will end up in the output directory for job #:', str(job_id)
	print "Urs: we are in directory: ", os.getcwd()
	print "and the params are: ", params
	#trace()
	return call_convnet(params) #  params['q_weights'], params['q_acts']

def write_params(param_file, params):
	
	epsW, epsB = params['steps'][0]/1000., params['steps'][1]/1000. # using a scaling factor to make parameters easier to 
	mom =  params['mom'] # 0.9
	wW1, wW2, wW3, wW4, wW5 = params['q_weights']
	aW1, aW2, aW3, aW4, aW5 = params['q_acts']
	# wW3, wW4, wW5 = 2, 2, 2
	# aW3, aW4, aW5 = 12, 12, 12 
	
	#wW1, aW1, wW2, aW2, wW5, aW5 = -1, 11, -3, 11, 15, 12

	f1=open(param_file, 'w+')

	print >>f1, '[conv1]'
	print >>f1, 'epsW=%f' %epsW
	print >>f1, 'epsB=%f' %epsB
	print >>f1, 'momW=%f' %mom
	print >>f1, 'momB=%f' %mom
	print >>f1, 'wc=0.000'
	print >>f1, 'weightWidth = %d,%d,0,0,0'   % (wW1, 15) # 0 24
	print >>f1, 'actWidth	 = %d,%d,0,0,0' % (aW1, 15) # 12 24
	print >>f1, ''
	print >>f1, '[conv2]'
	print >>f1, 'epsW=%f' %epsW
	print >>f1, 'epsB=%f' %epsB
	print >>f1, 'momW=%f' %mom
	print >>f1, 'momB=%f' %mom
	print >>f1, 'wc=0.000'
	print >>f1, 'weightWidth = %d,%d,0,0,0'   % (wW2, 15) # 0 24
	print >>f1, 'actWidth	 = %d,%d,0,0,0' % (aW2, 15) # 12 24
	print >>f1, ''
	print >>f1, '[local3]'
	print >>f1, 'epsW=%f' %epsW
	print >>f1, 'epsB=%f' %epsB
	print >>f1, 'momW=%f' %mom
	print >>f1, 'momB=%f' %mom
	print >>f1, 'wc=0.004'
	print >>f1, 'weightWidth = %d,%d,0,0,0'   % (wW3, 15) # 0 24
	print >>f1, 'actWidth	 = %d,%d,0,0,0' % (aW3-7, 15) # 12 24
	print >>f1, ''
	print >>f1, '[local4]'
	print >>f1, 'epsW=%f' %epsW
	print >>f1, 'epsB=%f' %epsB
	print >>f1, 'momW=%f' %mom
	print >>f1, 'momB=%f' %mom
	print >>f1, 'wc=0.004'
	print >>f1, 'weightWidth = %d,%d,0,0,0'   % (wW4, 15) # 0 24
	print >>f1, 'actWidth	 = %d,%d,0,0,0' % (aW4-7, 15) # 12 24
	print >>f1, ''
	print >>f1, '[fc10]'
	print >>f1, 'epsW=%f' %epsW
	print >>f1, 'epsB=%f' %epsB
	print >>f1, 'momW=%f' %mom
	print >>f1, 'momB=%f' %mom
	print >>f1, 'wc=0.01'
	print >>f1, 'weightWidth = %d,%d,0,0,0'   % (wW5, 15) # 0 24
	print >>f1, 'actWidth	 = %d,%d,0,0,0' % (aW5, 15) # 12 24
	print >>f1, ''
	print >>f1, '[logprob]'
	print >>f1, 'coeff        = 1'
	print >>f1, ''
	print >>f1, '[rnorm1]'
	print >>f1, 'scale        = 0.001'
	print >>f1, 'pow          = 0.75'
	print >>f1, 'minDiv       = 2'
	print >>f1, ''
	print >>f1, '[rnorm2]'
	print >>f1, 'scale        = 0.001'
	print >>f1, 'pow          = 0.75'
	print >>f1, 'minDiv       = 2'

	f1.close()

def write_layers(layer_file, params):

	f1=open(layer_file, 'w+')

	print >>f1, '[data]'
	print >>f1, 'type=data'
	print >>f1, 'dataIdx=0'
	print >>f1, ''

	print >>f1, '[labels]'
	print >>f1, 'type=data'
	print >>f1, 'dataIdx=1'
	print >>f1, ''

	print >>f1, '[conv1]'
	print >>f1, 'type=conv'
	print >>f1, 'inputs=data'
	print >>f1, 'channels=3'
	print >>f1, 'filters=64'
	print >>f1, 'padding=2'
	print >>f1, 'stride=1'
	print >>f1, 'filterSize=5'
	print >>f1, 'initW=0.0001'
	print >>f1, 'sumWidth=4'
	print >>f1, 'sharedBiases=1'
	print >>f1, 'gpu=0'
	print >>f1, 'neuron=' + params['nonlins'] 
	print >>f1, ''

	print >>f1, '[pool1]'
	print >>f1, 'type=pool'
	print >>f1, 'pool=max'
	print >>f1, 'inputs=conv1'
	print >>f1, 'sizeX=3'
	print >>f1, 'stride=2'
	print >>f1, 'channels=64'
	print >>f1, ''

	print >>f1, '[rnorm1]'
	print >>f1, 'type=cmrnorm'
	print >>f1, 'inputs=pool1'
	print >>f1, 'channels=64'
	print >>f1, 'size=9'
	print >>f1, ''

	print >>f1, '[conv2]'
	print >>f1, 'type=conv'
	print >>f1, 'inputs=rnorm1'
	print >>f1, 'filters=64'
	print >>f1, 'padding=2'
	print >>f1, 'stride=1'
	print >>f1, 'filterSize=5'
	print >>f1, 'channels=64'
	print >>f1, 'initW=0.01'
	print >>f1, 'sumWidth=2'
	print >>f1, 'sharedBiases=1'
	print >>f1, 'neuron=' + params['nonlins'] 
	print >>f1, ''

	print >>f1, '[rnorm2]'
	print >>f1, 'type=cmrnorm'
	print >>f1, 'inputs=conv2'
	print >>f1, 'channels=64'
	print >>f1, 'size=9'
	print >>f1, ''

	print >>f1, '[pool2]'
	print >>f1, 'type=pool'
	print >>f1, 'pool=max'
	print >>f1, 'inputs=rnorm2'
	print >>f1, 'sizeX=3'
	print >>f1, 'stride=2'
	print >>f1, 'channels=64'
	print >>f1, ''

	print >>f1, '[local3]'
	print >>f1, 'type=local'
	print >>f1, 'inputs=pool2'
	print >>f1, 'filters=64'
	print >>f1, 'padding=1'
	print >>f1, 'stride=1'
	print >>f1, 'filterSize=3'
	print >>f1, 'channels=64'
	print >>f1, 'neuron=relu'
	print >>f1, 'initW=0.04'
	print >>f1, ''

	print >>f1, '[local4]'
	print >>f1, 'type=local'
	print >>f1, 'inputs=local3'
	print >>f1, 'filters=32'
	print >>f1, 'padding=1'
	print >>f1, 'stride=1'
	print >>f1, 'filterSize=3'
	print >>f1, 'channels=64'
	print >>f1, 'neuron=relu'
	print >>f1, 'initW=0.04'
	print >>f1, ''

	print >>f1, '[fc10]'
	print >>f1, 'type=fc'
	print >>f1, 'outputs=10'
	print >>f1, 'inputs=local4'
	print >>f1, 'initW=0.01'
	print >>f1, ''

	print >>f1, '[probs]'
	print >>f1, 'type=softmax'
	print >>f1, 'inputs=fc10'
	print >>f1, ''

	print >>f1, '[logprob]'
	print >>f1, 'type=cost.logreg'
	print >>f1, 'inputs=labels,probs'
	print >>f1, 'gpu=0'

	f1.close()

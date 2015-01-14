# Here are some commands that I run spearmint with

../../spearmint/spearmint/bin/spearmint config.pb --driver=local --method=GPEIOptChooser --method-args=noiseless=0

../../spearmint/spearmint/bin/spearmint config.pb --driver=local --method=GPEIOptChooser --method-args=noiseless=0 --max-concurrent=2 -w --port=50001

../../spearmint/spearmint/bin/spearmint config.pb --driver=local --method=GPEIperSecChooser --method-args=noiseless=0 --polling-time=20 --max-concurrent=2 -w --port=50000

../../spearmint/spearmint/bin/cleanup ~/code/cuda-convnet2/spearmint/

# the way the current session (that I am about to start over) was running I got a best:
Minimum Value: 0.16705
Best Params:
epochs:[200],
q_weights:[-1 -4 -5 -6 -3],
q_acts:[11 11 3 4 11] # 3=10-7  4=11-7
# after running 500 models, but after 100 or so it should be pretty close. 
# Parameter ranges: 
Name: 	epochs
Type: 	INT
Size: 	1
Min: 	20.0
Max: 	200.0

Name: 	q_weights
Type: 	INT
Size: 	5
Min: 	-6.0
Max: 	-1.0

Name: 	q_acts
Type: 	INT
Size: 	5
Min: 	3.0
Max: 	11.0
# Here are some commands that I run spearmint with

../../../spearmint/spearmint/bin/spearmint spear_config.pb --driver=local --method=GPEIOptChooser --method-args=noiseless=0

../../../spearmint/spearmint/bin/spearmint spear_config.pb --driver=local --method=GPEIperSecChooser --method-args=noiseless=0 --polling-time=20 --max-concurrent=2 -w --port=50000

../../spearmint/spearmint/bin/cleanup ~/code/cuda-convnet2/spearmint/


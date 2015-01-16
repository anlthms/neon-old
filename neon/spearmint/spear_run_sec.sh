#!/bin/bash
../../../spearmint/spearmint/bin/spearmint spear_config.pb --driver=local --method=GPEIperSecChooser --method-args=noiseless=0 --polling-time=20 --max-concurrent=2 -w --port=50000
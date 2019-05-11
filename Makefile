loop:
	while ./virtual_skylight.py | tee -a virtual_skylight.log; do sleep 300s; done

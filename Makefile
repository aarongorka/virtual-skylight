loop:
	while ./virtual_skylight.py; do sleep 60; done

build:
	docker-compose build virtual-skylight

run:
	docker-compose run --rm virtual-skylight ./virtual_skylight.py

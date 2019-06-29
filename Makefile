loop:
	while ./virtual_skylight.py; do sleep 60; done

build:
	docker-compose build virtual-skylight

run:
	docker-compose run --rm virtual-skylight ./virtual_skylight.py --quiet

testTerm:
	docker-compose run --rm virtual-skylight python3 -c 'from fabulous import utils; term = utils.TerminalInfo(); print(term.width); print(term.height)'

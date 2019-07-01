#!/usr/bin/env python3
import yeelight
import sys
from virtual_skylight import load_config
import logging

logging.basicConfig(level=logging.INFO)

try:
    bulbs = load_config()['bulbs']
    bulbs.sort(key=lambda x: x['position'])
except FileNotFoundError:
    bulbs = yeelight.discover_bulbs()
    bulbs.sort(key=lambda x: x['capabilities']['id'])
assert len(bulbs) > 0

if sys.argv[1] == 'one_off':
    bulb_objs = [ yeelight.Bulb(bulb['ip'], auto_on=True) for bulb in bulbs ]
    on_bulbs = [ bulb for bulb in bulb_objs if bulb.get_properties()['power'] == 'on' ]
    if not on_bulbs == []:
        on_bulbs[0].turn_off()
else:
    for bulb in bulbs:
        if sys.argv[1] == 'off':
          this_bulb = yeelight.Bulb(bulb['ip'], auto_on=True)
          this_bulb.turn_off()
        else:
          print(bulb['ip'])
          this_bulb = yeelight.Bulb(bulb['ip'], auto_on=True)
          this_bulb.set_hsv(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), duration=int(sys.argv[4]))

#!/usr/bin/env python
import io
import requests
import cv2
import numpy as np
import skimage.io
from skimage import exposure
import yeelight
import logging
import pickle as pickle
import click
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def get_cropped_image(cache=False, debug=False):
    if cache and Path('content.pkl').is_file():
        logging.debug('Using cache...')
        content = pickle.load(open('content.pkl', 'rb'))
    else:
        logging.debug('Not using cache...')
        r = requests.get('https://captiveye-kirribilli.qnetau.com/refresh/getshot.asp?refresh=1557436280637', headers={"Referer": "https://captiveye-kirribilli.qnetau.com/refresh/default_embed.asp"})
        content = r.content
        if cache:
            logging.debug('Saving cache...')
            save_object(content, 'content.pkl')
    
    image_file = io.BytesIO(content)
    imread = skimage.io.imread(image_file)
    print('shape: {}'.format(imread.shape))
    if debug:
        skimage.io.imshow(imread)
        skimage.io.show()
    height = imread.shape[0]
    width = imread.shape[1]
    second_crop = imread[0:int(height/3), 0:width]
    print('2nd shape: {}'.format(second_crop.shape))
    if debug:
        skimage.io.imshow(second_crop)
        skimage.io.show()
    
    image = second_crop
    return image


def enhance_image(image, debug=False):
    """
    Improves the range of the image more closely align night with black and daylight with white.

    Reference: https://scikit-image.org/docs/dev/user_guide/transforming_image_data.html
    """

    better_contrast = exposure.rescale_intensity(image, in_range=(30, 200))
    if debug:
        skimage.io.imshow(better_contrast)
        skimage.io.show()
    return better_contrast


def get_dominant_colour(image, debug=False):
    """magic from https://stackoverflow.com/a/43111221/2640621"""

    average = [int(x) for x in image.mean(axis=0).mean(axis=0)]
    print(f'average: {average}')
    pixels = np.float32(image.reshape(-1, 3))
    
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = [int(x) for x in palette[np.argmax(counts)]]
    
    print('red: {red} green: {green} blue: {blue}'.format(red=dominant[0], green=dominant[1], blue=dominant[2]))
    return dominant


def rgb_to_hsv(r, g, b):
    """https://www.w3resource.com/python-exercises/math/python-math-exercise-77.php"""

    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v


def chunk_image_by_bulb(image):
    bulbs = range(1, 7)
    len(bulbs)


@click.command()
@click.option('--debug', is_flag=True)
@click.option('--cache/--no-cache')
def set_all_bulbs_to_sky(debug, cache):
    bulbs = []
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    image = get_cropped_image(cache=cache, debug=debug)
    better_image = enhance_image(image, debug=debug)
    rgb = get_dominant_colour(better_image, debug=debug)
    h, s, v = rgb_to_hsv(*rgb)
    print(f'hsv: {h} {s} {v}')
    bulbs = yeelight.discover_bulbs()  # TODO: set the colour of each bulb individually from a chunk of the image
    for bulb in bulbs:
        print(bulb['ip'])
        if v == 0:
            this_bulb = yeelight.Bulb(bulb['ip'], auto_on=True)
            this_bulb.turn_off()
        else:
            this_bulb = yeelight.Bulb(bulb['ip'], auto_on=True)
            this_bulb.set_hsv(h, s, v, duration=int(15000))


if __name__ == '__main__':
    set_all_bulbs_to_sky()

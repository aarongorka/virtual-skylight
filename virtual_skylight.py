#!/usr/bin/env python3
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
import backoff
from fabulous import image as fab_image
from fabulous import text
from fabulous.color import fg256
import imageio

logging.basicConfig(level=logging.INFO)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


exceptions = (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout)
@backoff.on_exception(backoff.expo, exceptions)
def get_image(cache=False, debug=False):
    if cache and Path('content.pkl').is_file():
        logging.debug('Using cache...')
        content = pickle.load(open('content.pkl', 'rb'))
    else:
        logging.debug('Not using cache...')
        r = requests.get('https://captiveye-kirribilli.qnetau.com/refresh/getshot.asp?refresh=1557436280637', headers={"Referer": "https://captiveye-kirribilli.qnetau.com/refresh/default_embed.asp"})
        r.raise_for_status()
        content = r.content
        if cache:
            logging.debug('Saving cache...')
            save_object(content, 'content.pkl')
    
    image_file = io.BytesIO(content)
    imread = skimage.io.imread(image_file)
    logging.info('Displaying original image:')
    print_scimage(imread)
    logging.debug('shape: {}'.format(imread.shape))
    if debug:
        skimage.io.imshow(imread)
        skimage.io.show()
    image = imread
    return image

def crop_image(image, debug=False):
    height = image.shape[0]
    width = image.shape[1]
    first_crop = image[0:int(height/2.29), 0:width]
    logging.debug('2nd shape: {}'.format(first_crop.shape))
    logging.info('Displaying cropped image:')
    print_scimage(first_crop)
    if debug:
        skimage.io.imshow(first_crop)
        skimage.io.show()
    
    image = first_crop
    return image

def crop_image_more(cropped_image, debug=False):
    height = cropped_image.shape[0]
    width = cropped_image.shape[1]
    second_crop = cropped_image[0:height, int(width * 0.925):int(width * 0.95)]
    logging.info('Displaying more cropped image:')
    print_scimage(second_crop)
    if debug:
        skimage.io.imshow(second_crop)
        skimage.io.show()
    return second_crop


def alt_crop_image_more(cropped_image, debug=False):
    height = cropped_image.shape[0]
    width = cropped_image.shape[1]
    second_crop = cropped_image[0:height, int(width * 0.5):width]
    logging.info('Displaying more cropped image:')
    print_scimage(second_crop)
    if debug:
        skimage.io.imshow(second_crop)
        skimage.io.show()
    return second_crop


def print_scimage(image):
    image_file = io.BytesIO()
    imageio.imwrite(image_file, image, format='png')
    print(fab_image.Image(image_file))


def enhance_image(image, debug=False):
    """
    Improves the range of the image more closely align night with black and daylight with white.

    Reference: https://scikit-image.org/docs/dev/user_guide/transforming_image_data.html
    """

    better_exposure = exposure.rescale_intensity(image, in_range=(0, 150))  # apply some contrast
    better_gamma = exposure.adjust_gamma(better_exposure, 2)
    final_adjustment = better_gamma

    if debug:
        skimage.io.imshow(final_adjustment)
        skimage.io.show()
    logging.info('Displaying enhanced image:')
    print_scimage(final_adjustment)
    if debug:
        skimage.io.imshow(final_adjustment)
        skimage.io.show()
    return final_adjustment


def get_dominant_colour(image, debug=False):
    """magic from https://stackoverflow.com/a/43111221/2640621"""

    average = [int(x) for x in image.mean(axis=0).mean(axis=0)][:3]
    html = "#{0:02x}{1:02x}{2:02x}".format(average[0], average[1], average[2])
    logging.info(fg256(html, f'average: {average}'))

    pixels = np.float32(image.reshape(-1, 4))
    
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
#    logging.info(f'palette: {palette}')

    filtered_palette = [x for x in palette if x[3] > 200]
    sorted_palette_dominant = np.sort([int(x) for x in filtered_palette[0]])

    dominant = [int(x) for x in palette[np.argmax(counts)]]
    
    html = "#{0:02x}{1:02x}{2:02x}".format(dominant[0], dominant[1], dominant[2])
    logging.info(fg256(html, f'dominant: {dominant}'))

    html = "#{0:02x}{1:02x}{2:02x}".format(sorted_palette_dominant[0], sorted_palette_dominant[1], sorted_palette_dominant[2])
    logging.info(fg256(html, f'sorted_palette_dominant: {sorted_palette_dominant}'))

    return sorted_palette_dominant[0], sorted_palette_dominant[1], sorted_palette_dominant[2]
#    return dominant[0], dominant[1], dominant[2]
#    return average[0], average[1], average[2]


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
    return int(h), int(s), int(v)


def get_hsv_by_bulb(image, bulbs, debug=False):
    number_of_chunks = len(bulbs)
    height = image.shape[0]
    width = image.shape[1]
    chunk_height = height / number_of_chunks
    for i, bulb in enumerate(bulbs):
        chunk_begin = int(chunk_height * i)
        chunk_end = int(chunk_begin + chunk_height)
        image_chunk = image[chunk_begin:chunk_end, 0:width]  # simple horizontal slices for now
        logging.info(f'Displaying chunk {i}:')
        print_scimage(image_chunk)
        if debug:
            skimage.io.imshow(image_chunk)
            skimage.io.show()
        r, g, b = get_dominant_colour(image_chunk)
        h, s, v = rgb_to_hsv(r, g * 0.8, b)  # slightly decrease greenness
        yield h, s, v, bulb


@click.command()
@click.option('--debug', is_flag=True)
@click.option('--cache/--no-cache')
@click.option('--off', is_flag=True)
@click.option('--dry-run', is_flag=True)
def set_all_bulbs_to_sky(debug, cache, off, dry_run):
    print(text.Text("Virtual Skylight", shadow=True, skew=5))
    bulbs = []
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    if off:
        bulbs = yeelight.discover_bulbs()
        for bulb in bulbs:
            this_bulb = yeelight.Bulb(bulb['ip'], auto_on=True)
            this_bulb.turn_off()
            logging.info('Goodbye!')
        return

    image = get_image(cache=cache, debug=debug)
    masked_image = apply_mask(image, debug=debug)
    cropped_image = crop_image(masked_image, debug=debug)
    alt_cropped_image = alt_crop_image_more(cropped_image, debug=debug)
    better_image = enhance_image(alt_cropped_image, debug=debug)
#    more_cropped_image = crop_image_more(cropped_image, debug=debug)
    final_image = better_image

    bulbs = yeelight.discover_bulbs()  # TODO: provide order to bulbs through options
    bulbs.sort(key=lambda x: x['capabilities']['id'])
    bulbs_and_hsvs = get_hsv_by_bulb(final_image, bulbs, debug=debug)
    for h, s, v, bulb in bulbs_and_hsvs:
        logging.info('Updating {}...'.format(bulb['ip']))
        if not dry_run:
            if v < 15:
                this_bulb = yeelight.Bulb(bulb['ip'], auto_on=True)
                this_bulb.turn_off()
            else:
                this_bulb = yeelight.Bulb(bulb['ip'], auto_on=True)
                this_bulb.set_hsv(h, s * 2, v, duration=int(15000))  # double saturation
    logging.info('Done.')


def apply_mask(image, debug=False):
    mask = skimage.io.imread("mask.png")
    print_scimage(mask)
    rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    result = cv2.subtract(rgba, mask)
    print_scimage(result)
    if debug:
        skimage.io.imshow(result)
        skimage.io.show()
    return result


if __name__ == '__main__':
    set_all_bulbs_to_sky()

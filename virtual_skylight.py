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
import imageio
import yaml
import time
import PIL

logging.basicConfig(level=logging.INFO)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


exceptions = (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout)
@backoff.on_exception(backoff.expo, exceptions)
def get_image(webcam, cache=False, debug=False, quiet=False):

    if cache and Path(webcam['cache']).is_file():
        logging.debug('Using cache...')
        content = pickle.load(open(webcam['cache'], 'rb'))
    else:
        logging.debug('Not using cache...')
        r = requests.get(webcam['url'], headers=webcam['headers'])
        r.raise_for_status()
        content = r.content
        if cache:
            logging.debug('Saving cache...')
            save_object(content, webcam['cache'])

    image_file = io.BytesIO(content)
    imread = skimage.io.imread(image_file)
    if not quiet:
        logging.info('Displaying original image:')
        print_scimage(imread)
    logging.debug('shape: {}'.format(imread.shape))
    if debug:
        skimage.io.imshow(imread)
        skimage.io.show()
    image = imread
    return image


def crop_image(image, dimensions, debug=False, quiet=False):
    first_crop = image[dimensions]
    logging.debug('2nd shape: {}'.format(first_crop.shape))
    if not quiet:
        logging.info('Displaying cropped image:')
        print_scimage(first_crop)
    if debug:
        skimage.io.imshow(first_crop)
        skimage.io.show()

    image = first_crop
    return image


def print_scimage(image):
    image_file = io.BytesIO()
    imageio.imwrite(image_file, image, format='png')
    print(FabImage(image_file))


def enhance_image(image, debug=False, quiet=False):
    """
    Improves the range of the image more closely align night with black and daylight with white.

    Reference: https://scikit-image.org/docs/dev/user_guide/transforming_image_data.html
    """

    better_exposure = exposure.rescale_intensity(image, in_range=(0, 150))  # apply some contrast
    better_gamma = exposure.adjust_gamma(better_exposure, 3)
    final_adjustment = better_gamma

    if debug:
        skimage.io.imshow(final_adjustment)
        skimage.io.show()
    if not quiet:
        logging.info('Displaying enhanced image:')
        print_scimage(final_adjustment)
    if debug:
        skimage.io.imshow(final_adjustment)
        skimage.io.show()
    return final_adjustment


def fg256(*args, **kwargs):
    """Workaround for fabulous imports breaking logging. Returns the uncoloured text if fg256 isn't imported."""

    try:
        return fab_fg256(*args, **kwargs)
    except NameError:
        return args[1]


def get_dominant_colour(image, debug=False, quiet=False):
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
    logging.debug(f'palette: {palette}')

    filtered_palette = [x for x in palette if x[3] > 200]
    logging.debug(f'filtered_palette: {filtered_palette}')
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


def get_hsv_by_bulb(image, bulbs, debug=False, quiet=False):
    number_of_chunks = len(bulbs)
    height = image.shape[0]
    width = image.shape[1]
    #height = image.size[1]
    #width = image.size[0]
    chunk_height = height / number_of_chunks
    for i, bulb in enumerate(bulbs):
        chunk_begin = int(chunk_height * i)
        chunk_end = int(chunk_begin + chunk_height)
        image_chunk = image[chunk_begin:chunk_end, 0:width]  # simple horizontal slices for now
        if not quiet:
            logging.info(f'Displaying chunk {i}:')
            print_scimage(image_chunk)
        if debug:
            skimage.io.imshow(image_chunk)
            skimage.io.show()
        r, g, b = get_dominant_colour(image_chunk)
        h, s, v = rgb_to_hsv(r, g * 0.8, b)  # slightly decrease greenness
        yield h, s, v, bulb


def apply_all_image_modifications(image, settings, debug=False, quiet=False):
    kwargs = {"debug": debug, "quiet": quiet}
    masked_image = apply_mask(image, settings['mask'], **kwargs)
    cropped_image = masked_image[settings['dimensions'][0]:settings['dimensions'][1], settings['dimensions'][2]:settings['dimensions'][3]]
    better_image = enhance_image(cropped_image, **kwargs)
    final_image = better_image
    return final_image


def apply_all_morning_image_modifications(image, debug=False, quiet=False):

    kwargs = {"debug": debug, "quiet": quiet}
    masked_image = apply_mask(image, 'morning_mask.png', **kwargs)
    cropped_image = masked_image[0:56, 0:300]
    better_image = enhance_image(cropped_image, **kwargs)
    final_image = better_image
    return final_image


def merge_images(morning, afternoon, debug=False, quiet=False):
    """https://stackoverflow.com/a/30228308/2640621"""

    morning_flipped = np.flipud(morning)  # https://stackoverflow.com/questions/9154120/how-can-i-flip-an-image-along-the-vertical-axis-with-python

    if not quiet:
        logging.info("Displaying flipped image:")
        print_scimage(morning_flipped)
    if debug:
        skimage.io.imshow(morning_flipped)
        skimage.io.show()

    imgs = [PIL.Image.fromarray(x) for x in [morning_flipped, afternoon]]

    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    logging.info(f'Min shape is {min_shape}')

    imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
#    imgs_comb = PIL.Image.fromarray(imgs_comb)

    if not quiet:
        logging.info('Displaying merged image:')
        print_scimage(imgs_comb)
    if debug:
        skimage.io.imshow(imgs_comb)
        skimage.io.show()
    return imgs_comb


@click.command()
@click.option('--debug', is_flag=True)
@click.option('--quiet', is_flag=True)
@click.option('--cache/--no-cache')
@click.option('--off', is_flag=True)
@click.option('--dry-run', is_flag=True)
@click.option('--alt-morning', is_flag=True)
def set_all_bulbs_to_sky(debug, quiet, cache, off, dry_run, alt_morning):
    if not quiet:
        # https://github.com/jart/fabulous/issues/17
        from fabulous.image import Image as FabImage
        from fabulous.text import Text as FabText
        from fabulous.color import fg256 as fab_fg256
        print(FabText("Virtual Skylight", shadow=True, skew=5))

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    bulbs = []
    start_time = time.time()

    try:
        bulbs = load_config()['bulbs']
        bulbs.sort(key=lambda x: x['position'])
    except FileNotFoundError:
        bulbs = yeelight.discover_bulbs()
        bulbs.sort(key=lambda x: x['capabilities']['id'])
    assert len(bulbs) > 0

    if off:
        for bulb in bulbs:
            this_bulb = yeelight.Bulb(bulb['ip'], auto_on=True)
            this_bulb.turn_off()
            logging.info('Goodbye!')
        return

    unixtime = int(time.time())

    webcams = {
        "morning": {
            "url": "http://weather.trevandsteve.com/snapshot1.jpg?x={unixtime}",
            "headers": {"Referer": "http://weather.trevandsteve.com/"},
            "mask": "morning_mask.png",
            "dimensions": [0, 313, 0, 639],
            "cache": "morning_content.pkl"
        },
        "alt_morning": {
            "url": "",
            "headers": {},
            "mask": "alt_morning_mask.png",
            "dimensions": [0, 56, 0, 300],
            "cache": "alt_morning_content.pkl"
        },
        "afternoon": {
            "url": "https://captiveye-kirribilli.qnetau.com/refresh/getshot.asp?refresh=1557436280637",
            "headers": {"Referer": "https://captiveye-kirribilli.qnetau.com/refresh/default_embed.asp"},
            "mask": "afternoon_mask.png",
            "dimensions": [0, 489, 0, 1920],
            "cache": "afternoon_content.pkl"
        }
    }

    if alt_morning:
        r = requests.get(f'https://api.deckchair.com/v1/camera/599d6375096641f2272bacf4/images?to={unixtime}')
        r.raise_for_status()
        identifier = r.json()['data'][0]['_id']
        webcams['alt_morning']['url'] = f'https://api.deckchair.com/v1/viewer/image/{identifier}?width=300&height=169&resizeMode=fill&gravity=Auto&quality=90&panelMode=false&format=jpg'

    kwargs = {"debug": debug, "quiet": quiet}

    afternoon_image = get_image(webcam=webcams['afternoon'], cache=cache, **kwargs)
    modified_afternoon_image = apply_all_image_modifications(image=afternoon_image, settings=webcams['afternoon'], **kwargs)
    if alt_morning:
        morning_image = get_image(webcam=webcams['alt_morning'], cache=cache, **kwargs)
        modified_morning_image   = apply_all_image_modifications(image=morning_image, settings=webcams['alt_morning'], **kwargs)
    else:
        morning_image = get_image(webcam=webcams['morning'], cache=cache, **kwargs)
        modified_morning_image   = apply_all_image_modifications(image=morning_image, settings=webcams['morning'], **kwargs)
    final_image = merge_images(morning=modified_morning_image, afternoon=modified_afternoon_image, **kwargs)

    elapsed = time.time() - start_time
    logging.info(f'Elapsed time: {elapsed}')
    sleep_time = (58 - elapsed) / len(bulbs)
    logging.info(f'Sleep time: {sleep_time}')
    sleeped = False

    bulbs_and_hsvs = get_hsv_by_bulb(final_image, bulbs, **kwargs)
    for h, s, v, bulb in bulbs_and_hsvs:
        if sleeped:  # skip the first sleep as we've probably already wasted time downloading images
            time.sleep(sleep_time)
        sleeped = True

        logging.info('Updating {}...'.format(bulb['ip']))
        if not dry_run:
            if v < 15:
                this_bulb = yeelight.Bulb(bulb['ip'], auto_on=True)
                try:
                    this_bulb.turn_off()
                except yeelight.main.BulbException:
                    logging.warning("Failed to turn off {}...".format(bulb['ip']))
            else:
                this_bulb = yeelight.Bulb(bulb['ip'], auto_on=True)
                try:
                    blended_h, blended_s, blended_v = blend_hsv(h, s, v, this_bulb)
                    this_bulb.set_hsv(blended_h, blended_s, blended_v, duration=15000)
                except yeelight.main.BulbException:
                    logging.critical("Failed to update {}...".format(bulb['ip']))
        logging.info('Updated {}.'.format(bulb['ip']))
    logging.info('Done.')


def blend_hsv(h: int, s: int, v: int, this_bulb: yeelight.Bulb):
    old_hsv = this_bulb.get_properties(requested_properties=['hue', 'sat', 'bright'])
    logging.info(f'Old hsv: {old_hsv}')
    if abs(int(old_hsv['hue']) - int(h)) > 128:
        blended_h: int = int((int(h) + int(old_hsv['hue'])) / 2)
    else:
        blended_h = h  # new/old value has probably rolled over to 0, don't bother calculating average
    blended_s: int = int((int(s) + int(old_hsv['sat']))    / 2)
    blended_v: int = int((int(v) + int(old_hsv['bright'])) / 2)
    logging.info(f'Blended hsv: {blended_h} {blended_s}, {blended_v}')
    return int(blended_h), int(blended_s), int(blended_v)


def apply_mask(image, mask_filename, debug=False, quiet=False):
    mask = skimage.io.imread(mask_filename)
    if not quiet:
        logging.info('Displaying mask:')
        print_scimage(mask)
    rgba_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2RGBA)
    rgba      = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    logging.info(f'Image shape: {rgba.shape}, mask shape: {rgba_mask.shape}')
    result = cv2.subtract(rgba, rgba_mask)
    if not quiet:
        logging.info('Displaying masked image:')
        print_scimage(result)
    if debug:
        skimage.io.imshow(result)
        skimage.io.show()
    return result


def load_config():
    with open('./config.yml') as handle:
        content = handle.read()
    config = yaml.safe_load(content)
    return config


if __name__ == '__main__':
    set_all_bulbs_to_sky()

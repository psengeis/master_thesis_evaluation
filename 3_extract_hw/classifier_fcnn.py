'''
This file can be used to test the models on input images. It includes features such as
1) Enabling CRF postprocessing
2) Enabling visualisations
3) Mean IoU calculations if a GT image is provided.

Run python 'classifier_fcnn.py -h' for more information

'''

import os
import argparse
import sys
import warnings

import cv2
from pathlib import Path
# import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from fcn_helper_function import weighted_categorical_crossentropy, IoU
from img_utils import getbinim, max_rgb_filter, get_IoU, getBinclassImg, mask2rgb, rgb2mask
from keras.engine.saving import load_model
from post import crf
from skimage import img_as_float
from skimage.color import gray2rgb

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# needed, as gpu to less memory...
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

BOXWDITH = 256
STRIDE = BOXWDITH - 10

ROOT = Path(__file__).parent.absolute()

# model = None


def classify(image):
    # global model

    # if model is None:
    model = load_model(str(Path(ROOT, 'models/fcnn_bin.h5')), custom_objects={
        'loss': weighted_categorical_crossentropy([0.4, 0.5, 0.1]), 'IoU': IoU})

    orgim = np.copy(image)
    image = img_as_float(gray2rgb(getbinim(image)))
    maskw = int((np.ceil(image.shape[1] / BOXWDITH) * BOXWDITH)) + 1
    maskh = int((np.ceil(image.shape[0] / BOXWDITH) * BOXWDITH))
    mask = np.ones((maskh, maskw, 3))
    mask2 = np.zeros((maskh, maskw, 3))
    mask[0:image.shape[0], 0:image.shape[1]] = image
    # print("classifying image...")
    for y in range(0, mask.shape[0], STRIDE):
        x = 0
        if (y + BOXWDITH > mask.shape[0]):
            break
        while (x + BOXWDITH) < mask.shape[1]:
            input = mask[y:y+BOXWDITH, x:x+BOXWDITH]
            std = input.std() if input.std() != 0 else 1
            mean = input.mean()
            mask2[y:y+BOXWDITH, x:x+BOXWDITH] = model.predict(
                np.array([(input-mean)/std]))[0]
            x = x + STRIDE
    return mask2[0:image.shape[0], 0:image.shape[1]]


def normalize(img):
    normalizedImg = cv2.normalize(img, np.zeros(img.shape[:2]), 0, 255, cv2.NORM_MINMAX)
    return normalizedImg

def tests(img):
    initial = img.copy()

    tested = normalize(img)

    cv2.imshow('initial', initial)
    cv2.imshow('tested', tested)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enableCRF", help="Use crf for postprocessing", action="store_true")
    parser.add_argument(
        "--input_image", help="input image file name", required=False)
    parser.add_argument(
        "--input_folder", help="input folder", required=False)

    args = parser.parse_args()

    images = None

    if args.input_image is not None:
        path = Path(args.input_image)
        assert path.is_file()
        images = [path]
    elif args.input_folder is not None:
        path = Path(args.input_folder)
        assert path.is_dir()
        images = path.glob('**/*.*')

    assert images is not None

    for img_path in images:

        # initial_img = io.imread(str(img_path))
        initial_img = cv2.imread(str(img_path))
        initial_img = normalize(initial_img)
        # tests(initial_img)
        # continue

        five_percent_height = int(initial_img.shape[0]/20)
        five_percent_width = int(initial_img.shape[1]/20)

        # take a selection of the image, if known where to search for
        # initial_img = initial_img[five_percent_height * 7:five_percent_height * 11,
        #                           0:five_percent_width * 12,
        #                           :]
        initial_img = np.ascontiguousarray(initial_img)

        # calculate scaling if needed - otherwise no upscale
        img_scale = min(
            1, min((1200/initial_img.shape[0]), (800/initial_img.shape[1])))

        classified = classify(initial_img)

        if args.enableCRF:
            crf_result = crf(initial_img, classified)
        else:
            crf_result = None

        if args.enableCRF:

            handwriting_mask = np.zeros((crf_result.shape))
            handwriting_mask[:, :][np.where(
                (crf_result[:, :] == [0, 0, 2]).all(axis=2))] = [0, 1, 0]

            handwriting_mask = cv2.resize(handwriting_mask, None,
                                          fx=img_scale, fy=img_scale)
            cv2.imshow('handwriting?', handwriting_mask)

        init_shrink = cv2.resize(initial_img, None,
                                 fx=img_scale, fy=img_scale)

        fcn_shrink = cv2.resize(max_rgb_filter(classified), None,
                                fx=img_scale, fy=img_scale)

        cv2.imshow('1', init_shrink)
        cv2.imshow('2', fcn_shrink)

        if args.enableCRF:
            crf_shrink = cv2.resize(crf_result, None,
                                    fx=img_scale, fy=img_scale)

            cv2.imshow('3', mask2rgb(crf_shrink))

        cv2.waitKey()
        cv2.destroyAllWindows()

import cv2
import numpy as np
from pathlib import Path
from classifier_fcnn import classify
from post import crf as postprocess
from color_sep import separate_by_color
from tqdm import tqdm


def extract(mode, img):

    extr = None
    if mode == 'nn':
        x = classify(img)
        extr = postprocess(img, x)
        handwriting_mask = np.zeros((extr.shape))
        handwriting_mask[:, :][np.where(
            (extr[:, :] == [0, 0, 2]).all(axis=2))] = [0, 1, 0]

        img_inv = cv2.bitwise_not(img)
        handwriting_mask = cv2.cvtColor(
            handwriting_mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        hw = cv2.bitwise_or(img_inv, img_inv, mask=handwriting_mask)
        extr = [cv2.bitwise_not(hw)]

    if mode == 'color':
        extr = separate_by_color(img)

    return extr


def run_evaluation(files, mode):

    for f in tqdm(files):
        img = cv2.imread(str(f))

        if mode == 'refs':
            inv_extr = cv2.bitwise_not(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inv_gray = cv2.bitwise_not(gray)

            _, tresholded = cv2.threshold(
                inv_gray, 125, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

            result = cv2.bitwise_or(inv_extr, inv_extr, mask=tresholded)

            if cv2.countNonZero(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)) > 500:
                # cv2.imshow('initial', img)
                # cv2.imshow('extracted', cv2.bitwise_not(result))
                # k = cv2.waitKey()
                # cv2.destroyAllWindows()

                t_file = Path(target, f'{f.stem}.png')
                cv2.imwrite(str(t_file), cv2.bitwise_not(result))

        else:
            extractions = extract(mode, img)

            for idx, extr in enumerate(extractions):

                inv_extr = cv2.bitwise_not(extr)
                gray = cv2.cvtColor(extr, cv2.COLOR_BGR2GRAY)
                inv_gray = cv2.bitwise_not(gray)

                _, tresholded = cv2.threshold(
                    inv_gray, 50, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

                result = cv2.bitwise_or(inv_extr, inv_extr, mask=tresholded)

                if cv2.countNonZero(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)) > 500:

                    # cv2.imshow('initial', img)
                    # cv2.imshow('extracted', extr)
                    # k = cv2.waitKey()
                    # cv2.destroyAllWindows()
                    # if k == ord('s'):
                    t_file = Path(target, f'{f.stem}_{str(idx)}.png')
                    cv2.imwrite(str(t_file), extr)


if __name__ == '__main__':

    target = Path('_output/3_extraction')
    target.mkdir(exist_ok=True)

    source = Path('_output/2_detected_objects/signs')

    files = [f for f in source.glob('*.png')]  # [:30]

    target.mkdir(exist_ok=True)
    run_evaluation(files, 'color')
    feature_target = Path(target.parent, '3_color')
    target.rename(feature_target)

    target.mkdir(exist_ok=True)
    run_evaluation(files, 'nn')
    feature_target = Path(target.parent, '3_neural')
    target.rename(feature_target)

    source = Path('_output/2_detected_objects/refs')

    files = [f for f in source.glob('*_mask_sign.png')]

    target.mkdir(exist_ok=True)
    run_evaluation(files, 'refs')
    feature_target = Path(target.parent, '3_ref_extr')
    target.rename(feature_target)

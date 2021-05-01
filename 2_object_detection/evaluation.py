import cv2
import numpy as np
from pathlib import Path
from detect import detect
from tqdm import tqdm
from mrcnn.visualize import apply_mask, random_colors


source = Path('_output/1_sections_ocr')
source_references = Path('_output/0_extr_pages')

target = Path('_output/2_detected_objects')


def detect_objects(img):

    sign_det = detect(img, detection_mode='signatures')
    stamp_det = detect(img, detection_mode='stamps')

    return sign_det, stamp_det


def write_result(result, proj_name, doc_typ):
    global target

    if len(result) > 0:
        det = result[0]
        page = det['position']['page']
        block = det['sign_block']

        target_file = f'{proj_name}_{doc_typ}_{str(page)}.png'
        cv2.imwrite(str(Path(target, target_file)), block)
    else:
        target_file = Path(target, f'{proj_name}_{doc_typ}_failed.txt')
        target_file.write_text('failed')


def get_masked_img(img, mask, roi):

    y1, x1, y2, x2 = roi

    target = np.zeros(img.shape, dtype=np.uint8)
    target[:] = 255
    for c in range(0, 3):
        target[:, :, c] = np.where(mask == 1,
                                   img[:, :, c],
                                   target[:, :, c])

    cropped_mask = target[y1:y2, x1:x2]

    # Find all non-zero points (text)
    coords = cv2.findNonZero(cv2.cvtColor(cropped_mask, cv2.COLOR_BGR2GRAY))
    x, y, w, h = cv2.boundingRect(coords)
    crop = cropped_mask[y:y+h, x:x+w]

    return crop


def run_model_for_paper():
    global target
    global source

    debug_imgs = Path(target, 'debug')
    debug_imgs.mkdir(exist_ok=True)

    files = [f for f in source.glob('*.png')]

    mask_sign = Path(target, 'signs')
    mask_sign.mkdir(exist_ok=True)

    mask_stamps = Path(target, 'stamps')
    mask_stamps.mkdir(exist_ok=True)

    colors = random_colors(3)

    for f in tqdm(files):

        img = cv2.imread(str(f))
        sign, stamps = detect_objects(img)

        debug_img = img.copy()

        sign_masks = Path()

        for idx, roi in enumerate(sign['rois']):

            mask = sign['masks'][:, :, idx]
            masked = get_masked_img(img, mask, roi)
            mask_file = Path(mask_sign, f'{f.stem}_{str(idx)}_sign.png')
            cv2.imwrite(str(mask_file), masked)

            y1, x1, y2, x2 = roi
            debug_img = cv2.rectangle(debug_img, (x1, y1), (x2, y2),
                          colors[0], thickness=2)  # purple
            
            debug_img = apply_mask(debug_img, mask, colors[0])

        for idx, roi in enumerate(stamps['rois']):

            mask = stamps['masks'][:, :, idx]
            masked = get_masked_img(img, mask, roi)
            mask_file = Path(mask_stamps, f'{f.stem}_{str(idx)}_stamp.png')
            cv2.imwrite(str(mask_file), masked)

            # cv2.rectangle(debug_img, (x1, y1), (x2, y2),
            #               (0, 255, 0), thickness=2)  # greens

            debug_img = apply_mask(debug_img, mask, colors[1])

        debug_file = Path(debug_imgs, f'{f.stem}.png')
        cv2.imwrite(str(debug_file), debug_img)


def run_model_for_references():
    global target
    global source_references

    mask_refs = Path(target, 'refs')
    mask_refs.mkdir(exist_ok=True)

    ref_files = [f for f in source_references.glob('**/ref_*.png')]

    for f in tqdm(ref_files):

        img = cv2.imread(str(f))
        references = detect(img, detection_mode='references')

        for idx, roi in enumerate(references['rois']):

            class_id = references["class_ids"][idx]

            if class_id != 1:
                continue

            mask = references['masks'][:, :, idx]
            masked = get_masked_img(img, mask, roi)

            mask_file = Path(
                mask_refs, f'{f.parent.name}_{f.stem}_{str(idx)}_mask_sign.png')
            cv2.imwrite(str(mask_file), masked)


if __name__ == '__main__':

    target.mkdir(exist_ok=True)

    run_model_for_paper()
    run_model_for_references()

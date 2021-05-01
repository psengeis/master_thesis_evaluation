import cv2
import numpy as np
import re  # regex
import pytesseract
from textdistance import ratcliff_obershelp

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'


def read_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(
        img_rgb, lang='deu', config="-c \"tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz \"")
    return text


def get_difference(a, b):
    return ratcliff_obershelp.normalized_distance(a, b)


def find_page(search_in_images, pages_to_find, required_positions, start_from_end):
    pages = []

    # currently searching only for one page
    find_txt = read_image(pages_to_find[0])

    # template_test = pages_to_find[0].copy()
    # d = pytesseract.image_to_data(template_test, output_type=pytesseract.Output.DICT)
    # n_boxes = len(d['level'])
    # for i in range(n_boxes):
    #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #     cv2.rectangle(template_test, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imshow('found_words_template', template_test)

    if start_from_end:
        search_in_images = [f for f in reversed(search_in_images)]

    for idx, possible in enumerate(search_in_images):
        txt = read_image(possible)
        difference = get_difference(txt, find_txt)
        if difference < 0.25:

            # test = possible.copy()
            # d = pytesseract.image_to_data(test, output_type=pytesseract.Output.DICT)
            # n_boxes = len(d['level'])
            # for i in range(n_boxes):
            #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            #     cv2.rectangle(test, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # cv2.imshow('found_words_' + str(idx), test)

            pages.append({
                'idx': idx,
                'text': txt,
                'difference': difference
            })

        if difference < 0.1:
            break

    # found_min, found_max = min(d_found['top']), max(d_found['top'])
    # templ_min, templ_max = min(d_templ['top']), max(d_templ['top'])

    # cv2.waitKey()

    selected_item = None
    if len(pages) > 0:
        selected_item = sorted(pages, key=lambda k: k['difference'])[0]

    cropped_sections = []
    if selected_item is not None and selected_item['difference'] < 0.2:
        found_page = search_in_images[selected_item['idx']]
        for pos in required_positions:
            x1, y1, x2, y2 = pos
            h, w = found_page.shape[:2]

            x1 = int(w * x1/100)
            x2 = int(w * x2/100)
            y1 = int(h * y1/100)
            y2 = int(h * y2/100)

            # should be transformed in future, as not perfectly scanned would have different zoom-factor
            page_idx = selected_item['idx']

            if start_from_end:
                page_idx = len(search_in_images) - page_idx - 1

            # cpy = found_page.copy()
            # cv2.circle(cpy, (x1, y1), 3, (0, 255, 0), 5)
            # cv2.circle(cpy, (x1, y2), 3, (0, 255, 0), 5)
            # cv2.circle(cpy, (x2, y1), 3, (0, 255, 0), 5)
            # cv2.circle(cpy, (x2, y2), 3, (0, 255, 0), 5)

            # cpy = cv2.resize(cpy, (600, 900))

            # cv2.imshow('edges', cpy)
            # cv2.waitKey()
            # cv2.destroyWindow('edges')

            cropped = found_page[y1:y2, x1:x2]
            cropped_sections.append({
                'accuracy': 1-selected_item['difference'],
                'sign_block': cropped,
                'position':  {
                    'page': page_idx,
                    'x': x1,
                    'y': y1,
                    'w': x2-x1,
                    'h': y2-y1
                }
            })

    return cropped_sections

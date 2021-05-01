import cv2
import numpy as np


def separate_by_color(image):

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    filter_sat = 25

    # # Calculate threshold based on horizontal projection
    # inv = 255 - image

    # initial = np.sum(inv)
    # current = initial

    # filter_sat = 0
    # while current/initial > 0.2:

    #     filter_sat += 1
    #     # filter based on blue color
    #     mask_blue = cv2.inRange(hsv_img, (90, filter_sat, 0), (155, 255, 255))
    #     c1_filled_patches = cv2.bitwise_or(inv, inv, mask=mask_blue)
    #     current = np.sum(c1_filled_patches)

    # print(filter_sat)

    while True:
        mask_green = cv2.inRange(hsv_img, (30, filter_sat, 0), (90, 255, 255))
        mask_blue = cv2.inRange(hsv_img, (90, filter_sat, 0), (155, 255, 255))

        tmp1 = cv2.inRange(hsv_img, (0, filter_sat, 0), (40, 255, 255))
        tmp2 = cv2.inRange(hsv_img, (145, filter_sat, 0), (255, 255, 255))
        mask_red = cv2.bitwise_or(tmp1, tmp2)

        white = np.full_like(image, 255)

        c1_filled_patches = cv2.bitwise_or(image, image, mask=mask_blue)
        c1_white_patches = cv2.bitwise_or(
            white, white, mask=cv2.bitwise_not(mask_blue))
        blue_colored = c1_filled_patches + c1_white_patches

        c2_filled_patches = cv2.bitwise_or(image, image, mask=mask_green)
        c2_white_patches = cv2.bitwise_or(
            white, white, mask=cv2.bitwise_not(mask_green))
        green_colored = c2_filled_patches + c2_white_patches

        c3_filled_patches = cv2.bitwise_or(image, image, mask=mask_red)
        c3_white_patches = cv2.bitwise_or(
            white, white, mask=cv2.bitwise_not(mask_red))
        red_colored = c3_filled_patches + c3_white_patches

        # gray
        mask_color = cv2.bitwise_or(
            cv2.bitwise_or(mask_green, mask_blue), mask_red)
        mask_inv = cv2.bitwise_not(mask_color)

        g_filled_patches = cv2.bitwise_or(
            image, image, mask=cv2.bitwise_not(mask_color))
        g_white_patches = cv2.bitwise_or(white, white, mask=mask_color)
        gray_segments = g_filled_patches + g_white_patches

        result = []
        for seg in [gray_segments, blue_colored, green_colored, red_colored]:
            if seg.max() > 80:
                result.append(seg)

        return result

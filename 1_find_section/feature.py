import cv2
import numpy as np


def find_page(pages_to_search_from, pages_to_find, infos):

    signed_signature_images = []

    if len(pages_to_find) == 0:
        return signed_signature_images

    detector = cv2.ORB_create(1000)
    min_matches = 30

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=32)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    missing_findings = []
    missing_infos = []
    missing_detections = []

    for idx, page in enumerate(pages_to_find):
        required_section = page
        required_section = cv2.cvtColor(required_section, cv2.COLOR_RGBA2GRAY)
        missing_findings.append(required_section)
        missing_infos.append(infos[idx])
        missing_detections.append(
            detector.detectAndCompute(required_section, None))

    for idx, page in enumerate(pages_to_search_from):

        img2 = page.copy()
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        if len(kp2) < 2 or desc2 is None:
            # empty page - no features on it
            continue

        for req_idx, req_img in enumerate(missing_findings):

            img1 = req_img
            kp1, desc1 = missing_detections[req_idx]
            x1, y1, x2, y2 = missing_infos[req_idx]
            h, w = img2.shape[:2]

            x1 = int(w * x1/100)
            x2 = int(w * x2/100)
            y1 = int(h * y1/100)
            y2 = int(h * y2/100)
            w = (x2-x1)
            h = (y2-y1)

            if len(kp1) < 2 or desc1 is None:
                # empty page - no features on it
                continue

            matches = matcher.knnMatch(desc1, desc2, 2)

            ratio = 0.75
            good_matches = [m[0] for m in matches
                            if len(m) == 2 and m[0].distance < m[1].distance * ratio]

            matchesMask = np.zeros(len(good_matches)).tolist()

            if len(good_matches) > min_matches:

                src_pts = np.float32(
                    [kp1[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32(
                    [kp2[m.trainIdx].pt for m in good_matches])

                mtrx, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, 5.0)

                matchesMask = mask.ravel().tolist()

                accuracy = float(mask.sum()) / mask.size

                if accuracy > 0.30:
                    # transformation to make it straight
                    img_h, img_w, = img1.shape[:2]
                    pts = np.float32(
                        [[[0, 0]], [[0, img_h-1]], [[img_w-1, img_h-1]], [[img_w-1, 0]]])

                    dst = cv2.perspectiveTransform(pts, mtrx)

                    # for displaying the transformation

                    # transformed = cv2.polylines(
                    #     img2.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

                    # res = cv2.drawMatches(img1, kp1, transformed, kp2, good_matches, None,
                    #                       matchesMask=matchesMask,
                    #                       flags=2) # cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

                    # cv2.imshow("matches", cv2.resize(
                    #     res, None, fx=0.3, fy=0.3))
                    # cv2.waitKey()

                    edge_points = [pnt[0] for pnt in np.int32(dst)]

                    e_min_x = min(list(zip(*edge_points))[0])
                    e_max_x = max(list(zip(*edge_points))[0])
                    e_min_y = min(list(zip(*edge_points))[1])
                    e_max_y = max(list(zip(*edge_points))[1])

                    y_ratio = (e_max_y - e_min_y) / img2.shape[0]
                    x_ratio = (e_max_x - e_min_x) / img2.shape[1]

                    min_x = int(e_min_x + x1 * x_ratio)
                    min_y = int(e_min_y + y1 * y_ratio)

                    max_x = int(min_x + w * x_ratio)
                    max_y = int(min_y + h * y_ratio)

                    min_x = max(0, min_x)
                    min_y = max(0, min_y)

                    max_x = min(img2.shape[1], max_x)
                    max_y = min(img2.shape[0], max_y)

                    cropped = img2[min_y:max_y, min_x:max_x]

                    # warped = cv2.warpPerspective(img2, mtrx, img1.shape)
                    # warped = warped[y:y+h, x:x+w]
                    if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                        signed_signature_images.append({
                            'matches': len(good_matches),
                            'accuracy': accuracy,
                            'sign_block': cropped,  # vs. warped
                            'position':  {
                                'page': idx,
                                'x': min_x,
                                'y': min_y,
                                'w': int(w * x_ratio),
                                'h': int(h * y_ratio)
                            }
                        })

    # print('matches found: {}'.format(len(signed_signature_images)))

    signed_signature_images.sort(key=lambda x: x['matches'], reverse=True)

    return signed_signature_images

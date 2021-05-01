import numpy as np
import pdf2image
import cv2
import re
import pytesseract
from pathlib import Path
from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'

def get_as_images(file_path, dpi=200):

    filename = file_path.name
    file_extension = file_path.suffix

    file_extension = file_extension.lower()
    supported_file_types = ['pdf', 'bmp', 'dib ', 'jpeg', 'jpg', 'png']

    if file_extension[1:] not in supported_file_types:
        raise f'Only following file types are supported: {", ".join(supported_file_types)}'

    if file_extension == '.pdf':
        return read_pages_from_pdf(file_path, dpi)
    else:
        return [cv2.imread(str(file_path))]


def read_pages_from_pdf(file, dpi=200, size=None):
    pil_images = pdf2image.convert_from_path(file, dpi, size=size)

    cv2_images = []

    for idx, pil_image in enumerate(pil_images):
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        cv2_images.append(open_cv_image)

    return cv2_images


def get_angle(image):
    angle = 0
    try:
        res = pytesseract.image_to_osd(image)
        conf = float(
            re.search('(?<=Orientation confidence: )\d*[.,]?\d*', res).group(0))

        if conf >= 1:
            angle = int(
                re.search('(?<=Orientation in degrees: )\d+', res).group(0))
    except:
        pass

    return angle


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    https://stackoverflow.com/a/47248339
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def get_rotated(image):
    angle = get_angle(image)
    rotated = rotate_image(image, angle)
    return rotated


def run_page_extraction(folder):
    global target
    for folder in tqdm(folder):

        proj_name = folder.name

        proj_folder = Path(target, proj_name)
        proj_folder.mkdir(exist_ok=True)

        try:
            for f in folder.glob('*'):

                imgs = get_as_images(f)

                proj_path = Path(proj_folder)

                for idx, img in enumerate(imgs):
                    t_path = proj_path
                    if f.name.startswith('request.'):
                        t_path = Path(t_path, f'antr_sign_{idx:04d}.png')
                    elif f.name.startswith('request_gen.'):
                        t_path = Path(t_path, f'antr_gen_{idx:04d}.png')
                    elif f.name.startswith('billing.'):
                        t_path = Path(t_path, f'abr_sign_{idx:04d}.png')
                    elif f.name.startswith('billing_gen.'):
                        t_path = Path(t_path, f'abr_gen_{idx:04d}.png')
                    elif f.name.startswith('reference_'):
                        t_path = Path(t_path, f'ref_{idx:04d}.png')

                    cv2.imwrite(str(t_path), img)

        except:
            print(f'Error at {proj_name}')


# source = Path('_input_prod')
source = Path('_input_thesis')
target = Path('_output/0_extr_pages')

if __name__ == '__main__':

    folder = [f for f in source.glob('*')]

    target.mkdir(exist_ok=True)
    print('Page extraction:')
    run_page_extraction(folder)

    print('Preprocessing:')

    files_to_process = target.glob('**/*_sign_*.png')
    files_to_process = [f for f in files_to_process]

    for file in tqdm(files_to_process):
        img = cv2.imread(str(file))
        cv2.imwrite(str(file), cv2.fastNlMeansDenoisingColored(img, h=3))

    for file in tqdm(files_to_process):
        img = cv2.imread(str(file))
        cv2.imwrite(str(file), get_rotated(img))

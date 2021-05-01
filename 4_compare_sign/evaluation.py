import cv2
import numpy as np
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
from signverification import InitConfig, SigVerSiameseCNN
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from tfhelper import PlotGPUInfo, RestrictMaxGPUMemoryAllocation

PlotGPUInfo()
RestrictMaxGPUMemoryAllocation(maxGB=4)

source_doc = Path('_output/3_color')
source_ref = Path('_output/3_ref_extr')

target = Path('_output/4_comparison')
projects = Path(target, 'projects')
results = Path(target, 'results')


def scale_image(img, max_height, max_width):
    sign_y, sign_x = img.shape[:2]

    ratio = 1

    ratio_x = max_width / sign_x
    ratio_y = max_height / sign_y

    ratio = min(ratio_x, ratio_y)

    target_width = int(sign_x * ratio)
    target_height = int(sign_y * ratio)

    img = cv2.resize(img, (target_width, target_height),
                     interpolation=cv2.INTER_AREA)

    color = (255)
    ww = max_width
    hh = max_height

    padded = np.full((max_height, max_width), color, dtype=np.uint8)

    ht, wd = img.shape[:2]
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    padded[yy:yy+ht, xx:xx+wd] = img

    return padded, ratio


model = None


def get_model():
    global model
    if model is None:
        config = InitConfig(False)
        model = SigVerSiameseCNN(config)
        old_model = load_model('./4_compare_sign/saved_model')
        model.set_weights(old_model.get_weights())

    return model


def compare(img1, img2):
    model = get_model()

    img_1_mod = img_to_array(img1)
    img_1_mod = img_1_mod.reshape((1, 440, 220, -1))

    img2_mod = img_to_array(img2)
    img2_mod = img2_mod.reshape((1, 440, 220, -1))

    predicted = model.predict([[img_1_mod, img2_mod]])
    print(predicted)

    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    # cv2.waitKey()

    # validate against the threshold
    if (predicted[0][0] < 30):
        return True
    
    return False


def get_img(path, target_size):
    sign = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    sign = cv2.threshold(sign, 200, 255, cv2.THRESH_BINARY)[1]
    return scale_image(sign, target_size[0], target_size[1])[0]


def write_file(project_nr, file_1, file_2, img_1, img_2, target_folder):
    t_file = f'{project_nr}-{Path(file_1).stem}-{Path(file_2).stem}.png'
    x = cv2.vconcat([img_1, img_2])
    cv2.imwrite(str(Path(target_folder, t_file)), x)


def run_evaluation(folders):
    global results
    global target

    # target_folder = Path(target, 'matches')
    # target_folder.mkdir(exist_ok=True)

    for f in tqdm(folders):
        project_nr = f.name

        antr_signatures = [f for f in f.glob('antr_*.png')]
        abr_signatures = [f for f in f.glob('abr_*.png')]
        ref_signatures = [f for f in f.glob('ref_*.png')]

        antr_correct = False
        abr_correct = False

        target_size = (220, 440)

        for ref in ref_signatures:
            ref_sign = get_img(ref, target_size)

            if antr_correct == False:
                for antr in antr_signatures:

                    antr_sign = get_img(antr, target_size)

                    # write_file(project_nr, ref, antr, ref_sign,
                    #            antr_sign, target_folder)

                    antr_correct = compare(ref_sign, antr_sign)
                    if antr_correct == True:
                        break

            if abr_correct == False:
                for abr in abr_signatures:
                    abr_sign = get_img(abr, target_size)

                    # write_file(project_nr, ref, antr, ref_sign,
                    #            antr_sign, target_folder)

                    abr_correct = compare(ref_sign, abr_sign)
                    if abr_correct == True:
                        break

            if antr_correct == True and abr_correct == True:
                break

        print(f'{project_nr}: Request = {antr_correct} // Billing = {abr_correct}')


def order_images(files_page, files_ref):
    global source_doc
    global source_ref
    global projects

    projects.mkdir(exist_ok=True)

    proj_folder = Path(projects)
    proj_folder.mkdir(exist_ok=True)

    files = files_page.copy()
    files.extend(files_ref)

    proj_nr_anzahl = 4  # Thesis: 4 // Prod: 8

    grouped = [list(g) for k, g in groupby(
        files, key=lambda x: x.name[:proj_nr_anzahl])]

    for f in grouped:

        antraege = 0
        abrechnungen = 0
        references = 0

        proj_name = f[0].name[:proj_nr_anzahl]
        current_proj = Path(proj_folder, proj_name)
        current_proj.mkdir(exist_ok=True)

        for file in f:

            target_file = None
            file_part = file.name[proj_nr_anzahl+1:proj_nr_anzahl + 4]

            if file_part == 'ant':
                target_file = Path(current_proj, f'antr_{str(antraege)}.png')
                antraege = antraege + 1
            elif file_part == 'abr':
                target_file = Path(
                    current_proj, f'abr_{str(abrechnungen)}.png')
                abrechnungen = abrechnungen + 1
            elif file_part == 'ref':
                target_file = Path(current_proj, f'ref_{str(references)}.png')
                references = references + 1
            else:
                continue

            img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            scale_image(img, 440, 220)
            cv2.imwrite(str(target_file), img)


if __name__ == '__main__':

    target.mkdir(exist_ok=True)

    files_page = [f for f in source_doc.glob('*.png')]
    files_ref = [f for f in source_ref.glob('*.png')]

    order_images(files_page, files_ref)

    folders = [f for f in projects.glob('*')]
    run_evaluation(folders)

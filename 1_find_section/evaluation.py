import cv2
from pathlib import Path
from feature import find_page as find_by_feature
from ocr import find_page as find_by_ocr
from tqdm import tqdm

SAVE_CORRECT_PAGE = True
source = Path('_output/0_extr_pages')
target = Path('_output/1_find_main')


def find_signature_position(mode, search_in_images, pages_to_find, required_positions, start_from_end):

    extr = None
    if mode == 'feature':
        extr = find_by_feature(search_in_images, pages_to_find,
                               required_positions)

    if mode == 'ocr':
        extr = find_by_ocr(search_in_images, pages_to_find,
                           required_positions, start_from_end)

    return extr


def write_result(result, proj_name, doc_typ, pages):
    global target
    global SAVE_CORRECT_PAGE

    if len(result) > 0:
        det = result[0]
        page = det['position']['page']
        block = det['sign_block']

        # cv2.imshow('img', block)
        # cv2.waitKey()
        # return
        target_file = f'{proj_name}_{doc_typ}_{str(page)}.png'
        cv2.imwrite(str(Path(target, target_file)), block)

        if SAVE_CORRECT_PAGE == True:
            pages_folder = Path(target, 'pages')
            pages_folder.mkdir(exist_ok=True)

            page_file = Path(pages_folder, f'{proj_name}_{doc_typ}.png')
            cv2.imwrite(str(page_file), pages[page])

    else:
        target_file = Path(target, f'{proj_name}_{doc_typ}_failed.txt')
        target_file.write_text('failed')


def run_evaluation(folder, mode):
    global antr_pos
    global abr_pos

    for f in tqdm(folder):
        proj_name = f.name

        # Request
        antr_gen = [x for x in Path(f).glob('antr_gen_*.png')][antr_pos[0]]
        antr_files = f.glob('antr_sign_*.png')

        antr_gen = cv2.imread(str(antr_gen))
        antr_files = [cv2.imread(str(f)) for f in antr_files]

        # x1, y1, x2, y2 = antr_pos[1:]
        # h, w = antr_gen.shape[:2]
        # block = antr_gen[int(y1 * h/100):int(y2 * h/100),
        #                  int(x1 * w/100):int(x2 * w/100)]
        # cv2.imshow('temp_request', block)

        antr_res = find_signature_position(
            mode, antr_files, [antr_gen], [antr_pos[1:]], True)

        write_result(antr_res, proj_name, 'antr', antr_files)

        # Billing
        abr_gen = [x for x in Path(f).glob('abr_gen_*.png')][abr_pos[0]]
        abr_files = f.glob('abr_sign_*.png')

        abr_gen = cv2.imread(str(abr_gen))
        abr_files = [cv2.imread(str(f)) for f in abr_files]

        # x1, y1, x2, y2 = abr_pos[1:]
        # h, w = abr_gen.shape[:2]
        # block = abr_gen[int(y1 * h/100):int(y2 * h/100),
        #                  int(x1 * w/100):int(x2 * w/100)]
        # cv2.imshow('temp_abr', block)
        # cv2.waitKey()

        abr_res = find_signature_position(
            mode, abr_files, [abr_gen], [abr_pos[1:]], False)

        write_result(abr_res, proj_name, 'abr', abr_files)


if __name__ == '__main__':
    folder = [f for f in source.glob('*')]

    # thesis documents:
    antr_pos = [0, 2, 60, 98, 80]
    abr_pos = [0, 35, 48, 90, 80]

    # prod_documents:
    # antr_pos = [-1, 2, 55, 98, 75]
    # abr_pos = [0, 0, 36, 80, 51]

    target.mkdir(exist_ok=True)
    run_evaluation(folder, 'feature')
    feature_target = Path(target.parent, '1_sections_feature')
    target.rename(feature_target)

    target.mkdir(exist_ok=True)
    run_evaluation(folder, 'ocr')
    feature_target = Path(target.parent, '1_sections_ocr')
    target.rename(feature_target)

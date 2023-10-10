import cv2 as cv
import pandas as pd
import config as cf
from Backend import Backend


def read_data(filename, start_index=None, end_index=None):
    """
    Функция создания датасета
    :param filename: файл с информацией о датасете
    :param start_index: Индекс начала парсинга
    :param end_index: Индекс конца парсинга
    :return: List[]
    """
    df = pd.read_csv(filename)

    start = 0 if start_index is None else start_index
    end = df.shape[0] if end_index is None else end_index

    for i in range(start, end):
        video_name = df.loc[i]['attachment_id']
        info = (df.loc[i]['height'], df.loc[i]['width'], df.loc[i]['length'])

        result = get_frames(video_name + '.mp4', df.loc[i]['begin'], df.loc[i]['end'])


def get_frames(path2file, begin_frame, end_frame):
    """
    Функция обработки видео по кадрам
    :param path2file:
    :param begin_frame:
    :param end_frame:
    :return: List
    """
    video = cv.VideoCapture(path2file)

    if not video.isOpened():
        return []

    frame_number = 1

    frames_list = []

    while video.isOpened():
        fl, frame = video.read()

        if begin_frame <= frame_number <= end_frame:
            frames_list.append(cv.resize(cf.FRAME_SIZE, frame) / 255)

    return process_element(frames_list)


def process_element(element, result_size=(35, *cf.FRAME_SIZE, 3), variant='4d'):
    """
    Функция изменения размерности данных
    :param element: Элемент для обработки
    :param result_size: Требуемые размеры
    :param variant: {
        '3d': для обработки трехмерных элементов (ч/б)
        '4d': для обработки четырехмерных элементов (rgb)
    }
    :return: List
    """
    if len(element) < result_size[0]:
        # Увеличение кол-ва фреймов
        left_index, right_index = 0, len(element) - 1
        check = True

        while len(element) != result_size[0]:
            if check:
                pass
            else:
                pass

    elif len(element) > result_size[0]:
        # Уменьшение кол-ва фреймов
        left_index, right_index = 0, len(element) - 1
        check = True

        while len(element) != result_size[0]:
            if check:
                element[left_index] = Backend.frame_unification(element[left_index], element[left_index + 1])

                del element[left_index + 1]
                left_index += 1
                check = False
            else:
                element[right_index] = Backend.frame_unification(element[right_index], element[right_index - 1])

                del element[right_index - 1]
                right_index -= 1
                check = True

            if left_index >= right_index:
                left_index, right_index = 0, len(element) - 1
    return element

import cv2 as cv


def read_data(filename, start_index=None, end_index=None):
    """
    Функция создания датасета
    :param filename: файл с информацией о датасете
    :param start_index: Индекс начала парсинга
    :param end_index: Индекс конца парсинга
    :return: List[]
    """
    pass


def get_frames(path2file):
    """
    Функция обработки видео по кадрам
    :param path2file:
    :return: List
    """
    pass


def process_element(element, result_size=(20, 640, 360, 3), variant='4d'):
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
    pass
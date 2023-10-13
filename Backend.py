import numpy as np
import config as cf


class Backend:
    @staticmethod
    def frame_unification(array_1, array_2, variant='4d'):
        """
        Функция для объединения кадров
        :param array_1:
        :param array_2:
        :param variant: {
        '3d': для обработки трехмерных элементов (ч/б)
        '4d': для обработки четырехмерных элементов (rgb)
    }
        :return:
        """
        new_frame = []

        for y in range(cf.FRAME_SIZE[1]):
            h_lst = []
            for x in range(cf.FRAME_SIZE[0]):
                rgb_1, rgb_2 = array_1[y][x], array_2[y][x]

                new_rgb = ((rgb_1[0] + rgb_2[0]) / 2, (rgb_1[1] + rgb_2[1]) / 2, (rgb_1[2] + rgb_2[2]) / 2)
                h_lst.append(new_rgb)

            new_frame.append(np.array(h_lst))

        return np.array(new_frame)

    @staticmethod
    def frame_express(*frames):
        """
        Функция для поиска линейных уравнений и воссоздания нового кадра
        :param frames: массив кадров
        :return:
        """
        new_frame = []
        for y in range(cf.FRAME_SIZE[1]):
            h_lst = []
            for x in range(cf.FRAME_SIZE[0]):
                red_list = [(i + 1, frames[i - 1][x][y][0]) for i in range(1, 6)]
                green_list = [(i + 1, frames[i][x][y][1]) for i in range(1, 6)]
                blue_list = [(i + 1, frames[i][x][y][2]) for i in range(1, 6)]

                k, b = Backend.get_coefficients(red_list)
                new_red = max(k + b, 255)

                k, b = Backend.get_coefficients(green_list)
                new_green = max(k + b, 255)

                k, b = Backend.get_coefficients(blue_list)
                new_blue = max(k + b, 255)

                new_rgb = np.array((new_red, new_green, new_blue))
                h_lst.append(new_rgb)
            new_frame.append(np.array(h_lst))

        return np.array(new_frame)

    @staticmethod
    def get_coefficients(points):
        """
        Функция для поиска ближайшей ко всем точкам прямой
        :param points:
        :return:
        """
        size = len(points)
        # формируем и заполняем матрицу размерностью 2x2
        A = np.empty((2, 2))
        A[[0], [0]] = sum((points[0][i]) ** 2 for i in range(size))
        A[[0], [1]] = sum([pair[0] for pair in points])
        A[[1], [0]] = sum([pair[0] for pair in points])
        A[[1], [1]] = size
        # находим обратную матрицу
        A = np.linalg.inv(A)
        # формируем и заполняем матрицу размерностью 2x1
        C = np.empty((2, 1))
        C[0] = sum((points[0][i] * points[1][i]) for i in range(size))
        C[1] = sum((points[1][i]) for i in range(size))

        # умножаем матрицу на вектор
        ww = np.dot(A, C)

        return ww[1], ww[0]

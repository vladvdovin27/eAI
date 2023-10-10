import numpy as np
import config as cf


class Backend:
    @staticmethod
    def frame_unification(frame_1, frame_2, variant='4d'):
        """
        Функция для объединения кадров
        :param frame_1:
        :param frame_2:
        :param variant: {
        '3d': для обработки трехмерных элементов (ч/б)
        '4d': для обработки четырехмерных элементов (rgb)
    }
        :return:
        """
        array_1, array_2 = np.asarray(frame_1), np.asarray(frame_2)
        new_frame = []

        for y in range(cf.FRAME_SIZE[0]):
            h_lst = []
            for x in range(cf.FRAME_SIZE[1]):
                rgb_1, rgb_2 = array_1[y][x], array_2[y][x]

                new_rgb = ((rgb_1[0] + rgb_2[0]) / 2, (rgb_1[1] + rgb_2[1]) / 2, (rgb_1[2] + rgb_2[2]) / 2)
                h_lst.append(new_rgb)

            new_frame.append(np.array(h_lst))

        return np.array(new_frame)

    @staticmethod
    def get_coefficients(frame_1, frame_2):
        """
        Функция для получения коэффициентов изменения кадров
        :param frame_1:
        :param frame_2:
        :return:
        """
        array_1, array_2 = np.asarray(frame_1), np.asarray(frame_2)
        coefficients = []

        for y in range(cf.FRAME_SIZE[0]):
            h_lst = []
            for x in range(cf.FRAME_SIZE[1]):
                rgb_1, rgb_2 = array_1[y][x], array_2[y][x]

                k1 = (rgb_2[0] - rgb_1[0])
                b1 = rgb_2[0] - k1 * 2

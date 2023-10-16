import lst as lst
import tensorflow as tf
import keras
from keras.layers import *
import config as cf


def convoluted_part(input_layer):
    """
    Функция для создания сверточной части нейросети
    :param input_layer: Входной слой
    :return:
    """
    bn_0 = BatchNormalization()(input_layer)
    cv_0 = Conv2D(4, 3, activation='relu')(bn_0)
    cv_1 = Conv2D(16, 3, activation='relu')(cv_0)
    mp_0 = MaxPooling2D()(cv_1)
    bn_1 = BatchNormalization()(mp_0)
    cv_2 = Conv2D(32, 5, activation='relu')(bn_1)
    dr_0 = Dropout(0.2)(cv_2)
    cv_3 = Conv2D(64, 5, activation='relu')(dr_0)
    mp_1 = MaxPooling2D()(cv_3)
    bn_2 = BatchNormalization()(mp_1)
    cv_4 = Conv2D(64, 3, activation='relu')(bn_2)
    cv_5 = Conv2D(84, 3, activation='relu')(cv_4)
    mp_2 = MaxPooling2D()(cv_5)
    bn_3 = BatchNormalization()(mp_2)
    cv_6 = Conv2D(84, 3, activation='relu')(bn_3)
    dr_1 = Dropout(0.2)(cv_6)
    cv_7 = Conv2D(128, 3, activation='relu')(dr_1)
    mp_3 = MaxPooling2D()(cv_7)
    bn_4 = BatchNormalization()(mp_3)
    mp_4 = MaxPooling2D()(bn_4)

    return mp_4


def create_model(input_shape=cf.NN_INPUT_SIZE):
    """
    Функция для создания нейросети
    :param input_shape: Размеры входных данных
    :return:
    """
    inputs_list = []
    conv_outputs_list = []

    # Создание всех слоев входа для каждой сверточной части
    for _ in range(input_shape[0]):
        inputs_list.append(keras.Input(shape=[input_shape[1], input_shape[2], input_shape[3]]))

    # Создание всех сверточных частей и получение последнего слоя каждой из них
    for i in range(input_shape[0]):
        conv_outputs_list.append(convoluted_part(inputs_list[i]))

    connect_layer = concatenate([layer for layer in conv_outputs_list])

    rec_part = recurrent_part(connect_layer)

    lin_0 = Dense(128, activation='relu')(rec_part)
    output = Dense(cf.OUTPUT_SIZE[0], activation='softmax')(lin_0)

    return keras.Model(inputs=inputs_list, outputs=[output])


def recurrent_part(rec_layer):
    """
    Функция для создания рекуррентной
    :param rec_layer:
    :return:
    """
    r_bn_0 = BatchNormalization()(rec_layer)
    lstm_0 = LSTM(16, return_sequence=True, activation='relu')(r_bn_0)
    lstm_1 = LSTM(32, return_sequences=True, activation='relu')(lstm_0)
    r_bn_1 = BatchNormalization()(lstm_1)
    lstm_2 = LSTM(128, return_sequences=True, activation='relu')(r_bn_1)
    r_dr_0 = Dropout(0.25)(lstm_2)
    lstm_3 = LSTM(512, activation='relu')(r_dr_0)
    r_bn_2 = BatchNormalization()(lstm_3)

    return r_bn_2

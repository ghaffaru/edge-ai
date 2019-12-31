import cv2
import numpy as np


def handle_pose(output, input_shape):

    return None


def handle_text(output, input_shape):

    return None


def handle_car(output, input_shape):

    color = output['color'].flatten()
    car_type = output['type'].flatten()

    color_class = np.argmax(color)
    type_class = np.argmax(car_type)

    return color_class, type_class


def handle_output(model_type):

    if model_type == 'POSE':

        return handle_pose

    elif model_type == 'TEXT':

        return handle_text

    elif model_type == 'CAR_META':

        return handle_car

    else:

        return None


def preprocessing(input_image, height, width):

    image = cv2.resize(input_image, (width,height))

    image = image.transpose((2, 0, 1))

    image = image.reshape(1, 3, height, width)

    return image

"""Module for loading data from jpg files."""
import re
import cv2
from helpers import angle, build_vector

HEADERS = ["corners_count", "right_angle_counter", "parallel_sides_counter", "h_w_ratio", "file_path", "label"]

def load_properties_list(image_path):
    """Load properties of a shape from a jpg path."""
    file_path = image_path
    file_name = file_path.split('/')[-1]
    label = re.sub(r'\d', '', file_name)[:-4]

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    corners = cv2.goodFeaturesToTrack(img, 64, 0.3, 50)
    corners_count = len(corners)

    angle_list = []
    for i in range(len(corners)):
        j = (i+1) % len(corners)
        angle_list.append(angle(build_vector(*corners[i][0], *corners[j][0]), build_vector(*corners[j][0], *corners[(j+1) % len(corners)][0])))

    right_angle_counter = sum(87 < angle < 93 for angle in angle_list)

    side_list = [build_vector(*corners[i][0], *corners[(i+1) % len(corners)][0]) for i in range(len(corners)) for _ in range(2)]
    summarized_directional_coefficient_list = [(x, side_list.count(x)) for x in set(side_list)]
    parallel_sides_counter = sum(count == 2 or count == 4 for _, count in summarized_directional_coefficient_list)

    hight, width = max(corners, key=lambda c: c[0][1])[0][1] - min(corners, key=lambda c: c[0][1])[0][1], max(corners, key=lambda c: c[0][0])[0][0] - min(corners, key=lambda c: c[0][0])[0][0]
    h_w_ratio = round(hight/width, 1)

    return [corners_count, right_angle_counter, parallel_sides_counter, h_w_ratio, file_path, label]

def load_training_data_list(jpg_path_list):
    """Load training data from a list of jpg paths."""
    main_list = []
    for path_list_element in jpg_path_list:
        loaded_properties = load_properties_list(path_list_element)
        main_list.append(loaded_properties)
    return main_list

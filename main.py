"""Decision tree classifier for 2D shapes recognition."""
from colorama import Style
from load_data import load_training_data_list
from decision_tree import build_tree, draw_tree, print_tree, export_tree, print_leaf, classify

def print_input_data_element(element):
    """Prints the input data element in a readable format."""
    # print(f"{Style.DIM}label:", f"{Style.NORMAL}{element[6]}")
    # print(f"{Style.DIM}file_name:", f"{Style.NORMAL}{element[5]}")
    print(f"{Style.DIM}label:", f"{Style.NORMAL}{element[5]}")
    print(f"{Style.DIM}file_path:", f"{Style.NORMAL}{element[4]}")
    print(f"{Style.DIM}corners_count:", f"{Style.NORMAL}{element[0]}")
    print(f"{Style.DIM}right_angle_counter:", f"{Style.NORMAL}{element[1]}")
    print(f"{Style.DIM}parallel_sides_counter:", f"{Style.NORMAL}{element[2]}")
    print(f"{Style.DIM}h_w_ratio:", f"{Style.NORMAL}{element[3]}")

def print_classification(data, tree):
    """Prints the classification of the testing data."""
    for data_row_idx, data_row in enumerate(data):
        classification = classify(data_row, tree)
        printed_leaf = print_leaf(classification)
        prediction = list(printed_leaf.keys())[0]
        confidence = list(printed_leaf.values())[0]

        print(f"Testing data element #{data_row_idx}:")
        print_input_data_element(data_row)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence}")
        print(Style.RESET_ALL)

if __name__ == '__main__':
    training_default_jpg_path_list = (
      'train_images/circle01.jpg', 
      'train_images/circle02.jpg', 
      'train_images/elipse01.jpg', 
      'train_images/elipse02.jpg', 
      'train_images/rectangle01.jpg', 
      'train_images/rectangle02.jpg', 
      'train_images/rectangle03.jpg', 
      'train_images/rhombus01.jpg', 
      'train_images/square01.jpg', 
      'train_images/square02.jpg', 
      'train_images/triangle01.jpg', 
      'train_images/triangle02.jpg'
    )

    training_data = load_training_data_list(training_default_jpg_path_list)
    for training_data_element_idx, training_data_element in enumerate(training_data):
        print(f"Training data element #{training_data_element_idx}:")
        print_input_data_element(training_data_element)
        print()

    my_tree = build_tree(training_data)

    print_tree(my_tree)
    tree_drawing = draw_tree(my_tree)
    print()
    export_tree(tree_drawing)

    testing_default_jpg_path_list = (
      'input/test1.jpg',
      'input/test2.jpg', 
      'input/test3.jpg'
      )

    print()
    print('Do you want to add your own test data? (y/n)')
    answer = input()
    if answer == 'y':
        print()
        while True:
            print('Enter paths to you test data (separated by commas):')
            testing_data = load_training_data_list(input().replace(' ', '').split(','))
            print_classification(testing_data, my_tree)
    else:
        print()
        testing_data = load_training_data_list(testing_default_jpg_path_list)
        print('Classification for default testing data:')
        print_classification(testing_data, my_tree)

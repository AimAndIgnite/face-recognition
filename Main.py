import cv2
import numpy as np
from matplotlib import pyplot as plt 
import os


def get_path_list(root_path):
    # Obtains list of path directories from root path
    # Returns: list(s) containing the names of the sub-directories in the root directory

    train_root_path_content = []
    for i in os.listdir(root_path):
        train_root_path_content.append(i)
    return train_root_path_content
#--------------------------------------------------------------------------------------------

def get_class_names(root_path, train_names):
    # Obtains a list of training images path and a list of image classes id
    # Returns:
    #   - A list containing all image paths in the train directories
    #   - A list containing all image classes id

    train_images_path = []
    train_images_id = []
    for i, folder_name in enumerate(train_names):
        image_folder = root_path + '/' + folder_name
        for image_files in os.listdir(image_folder):
            image_path = image_folder + '/' + image_files
            train_images_path.append(image_path)
            train_images_id.append(i)       
    return train_images_path, train_images_id
#--------------------------------------------------------------------------------------------

def get_train_images_data(image_path_list):
    # Loads a list of train images from given path list
    # Returns:
    #  - A list containing all loaded training images

    loaded_images = []
    for images in image_path_list:
        # read_image = cv2.imread(images, 0)
        read_image = cv2.imread(images)
        loaded_images.append(read_image)
    return loaded_images
#--------------------------------------------------------------------------------------------

def detect_faces_and_filter(image_list, image_classes_list=None):
    # Detects any faces from a given image list and filters said face if 
    # the face on the given image is more or less than one
    # Returns
    #   - A list containing all filtered and cropped face images in grayscale
    #   - A list containing all filtered faces location saved in rectangle
    #   - A list containing all filtered image classes id

    filtered_and_cropped_face = []
    filtered_and_located_face = []
    filtered_id = []

    # Haar Cascade to detect face(s)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # detection process
    for i, images in enumerate(image_list):
        gray_image = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)
        
        # only process a single face in the picture
        if len(detected_faces) != 1:
            continue
        
        # face = x, y, w, h
        for x, y, w, h in detected_faces:
            # cut
            face_rect = gray_image[y:y+h, x:x+w]
            filtered_and_cropped_face.append(face_rect)
            filtered_and_located_face.append([x, y, w, h])
            #filter_id.append(i)  

            if image_classes_list != None:
                filtered_id.append(image_classes_list[i])
    return filtered_and_cropped_face, filtered_and_located_face, filtered_id
#--------------------------------------------------------------------------------------------

def train(train_face_grays, image_classes_list):
    # Creates and trains a classifier object
    # Returns
    #   - A classifier object after being trained with images of cropped faces
  
    face_detector = cv2.face.LBPHFaceRecognizer_create()
    face_detector.train(train_face_grays, np.array(image_classes_list))
    return face_detector
#--------------------------------------------------------------------------------------------

def get_test_images_data(test_root_path, image_path_list):
    # Loads a list of test images from a given path list
    # Returns
    #  - List containing all loaded test images
  
    test_image_list = []

    for images in image_path_list:
        test_image_read = cv2.imread(test_root_path + '/' + images)
        test_image_list.append(test_image_read)
    return test_image_list
#--------------------------------------------------------------------------------------------

def predict(classifier, test_faces_gray):
    # Predicts the test image 
    # Returns
    #   - List containing all prediction results from given test faces

    prediction_results = []
    for images in test_faces_gray:
        result, confidence = classifier.predict(images)
        prediction_results.append(result)
    return prediction_results
#--------------------------------------------------------------------------------------------

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    # Draws prediction results on the given test images
    # Returns
    #   - List containing all test images after being highlighted with prediction results

    draw_results = []
    for i, images in enumerate(test_image_list):
        prediction = predict_results[i]
        drawn = 0 
        # place rectangle and text only once for every test image
        for x,y,w,h in test_faces_rects:
            if drawn == 0:
                cv2.rectangle(images, (x,y), (x+w, y+h), (0, 255, 0), 2)
                text = train_names[prediction]
                cv2.putText(images, text, (x+5, y-10), 0, 0.5, (0,255,0, 2))
                '''
                Parameters:
                1 variable
                2 text to be written
                3 coordinates of text
                4 font face
                5 font size
                6 text color (bgr)
                '''
                drawn = 1
            else:
                continue
        draw_results.append(images)
    return draw_results
#--------------------------------------------------------------------------------------------

def combine_results(predicted_test_image_list):
       # Combines all predicted test image result into a single image
       # Returns
       #  - Array (ndarray) containing image data after being combined
    
    combined_images = predicted_test_image_list[0]
    for image in predicted_test_image_list[1:]:
        combined_images = np.hstack((combined_images, image))
    return combined_images
#--------------------------------------------------------------------------------------------

def show_result(image):
    # Shows the prediction image
        
    cv2.imshow("Final Results", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Modify value of train_root_path with the location of your training data root directory
    train_root_path = "dataset/train"
    
    train_names = get_path_list(train_root_path)
    image_path_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_image_list = get_train_images_data(image_path_list)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    classifier = train(train_face_grays, filtered_classes_list)

    # Modify value of test_image_path with the location of your testing data root directory
    test_root_path = "dataset/test"

    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path, test_names)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(classifier, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    final_image_result = combine_results(predicted_test_image_list)
    show_result(final_image_result)

import face_recognition
import pickle
import mediapipe as mp
import numpy as np
import cv2
import os


import matplotlib.pyplot as plt

def display_image(image_array):
    plt.imshow(image_array)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()


def get_jpg_files(folder_path):
    jpg_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpeg"):
            jpg_files.append(os.path.join(folder_path, file_name))
    return jpg_files

def get_face_bbox(frame):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        face_results = face_mesh.process(frame)
        
        face_landmarks=face_results.multi_face_landmarks
        h, w, _ = frame.shape
        for landmarks in face_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
    return x_min, y_min,x_max, y_max

def resize_image_with_aspect_ratio(image_array, target_height):
    # Calculate the scale factor to maintain the aspect ratio
    scale_factor = target_height / image_array.shape[0]
    # Resize the image with the calculated scale factor
    resized_image = cv2.resize(image_array, (int(image_array.shape[0] * scale_factor), int(image_array.shape[1] * scale_factor)))
    return resized_image

def detect_faces(image):
  
  x_min, y_min,x_max, y_max = get_face_bbox(image)
  
  cropped_face = image[y_min-80:y_max+80,x_min-80:x_max+80]
    
   
  return np.ascontiguousarray(cropped_face)

def generate_embeddings(image_path):
    image = face_recognition.load_image_file(image_path)
    cropped_image=detect_faces(image)
    img_shape=cropped_image.shape

    if(img_shape[0] < img_shape[1]):
        cropped_image=np.transpose(cropped_image, (1,0,2))

    cropped_image=resize_image_with_aspect_ratio(cropped_image,720)

    display_image(cropped_image)
    face_embeddings = face_recognition.face_encodings(cropped_image)[0]

    return face_embeddings, image_path.split('/')[-1].split('.')[0]


def generate_embeddings_pkl():
    face_paths=get_jpg_files('./Faces')
    
    FaceEncodings = []
    FaceNames = []
    for path in face_paths:
        try:
            embedding, name=generate_embeddings(path)
            FaceEncodings.append(embedding)
            FaceNames.append(name)
        except:
            print('face not identified')
            pass
    
    embeddings_dict = {'FaceEmbeddings_new': FaceEncodings, 'FaceNames': FaceNames}
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings_dict, f)


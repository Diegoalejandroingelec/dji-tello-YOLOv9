import face_recognition
import mediapipe as mp
import numpy as np



def get_true_index(bool_list,FaceNames):
    for idx, value in enumerate(bool_list):
        if value:
            return FaceNames[idx]
    return -1  # Return -1 if no true value is found


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



def detect_faces(image):
  
  x_min, y_min,x_max, y_max = get_face_bbox(image)
  
  cropped_face = image[y_min:y_max,x_min:x_max]
    
   
  return np.ascontiguousarray(cropped_face)


def identify_user(frame,FaceNames,FaceEncodings):
    try:
       cropped_face = detect_faces(frame.copy())
       unknown_face_encoding = face_recognition.face_encodings(cropped_face)[0]
        # Compare the faces with the encodings
       results = face_recognition.compare_faces(FaceEncodings, unknown_face_encoding, 0.55)
       index = get_true_index(results,FaceNames)

       if index == -1:
           print('not identified!')
           return False , '-'
       else:
           print(index)
           return True, index
    except:
        pass


# cap = cv2.VideoCapture(0)
# # te_ip= '192.168.10.1'
# # cap =  cv2.VideoCapture('udp://' + te_ip + ':11111')

# # Load the face encodings from the pickle file
# with open('embeddings.pkl', 'rb') as f:
#     FaceEmbeddings = loaded_embeddings_dict = pickle.load(f)

# FaceEncodings= FaceEmbeddings["FaceEmbeddings_new"]
# FaceNames = FaceEmbeddings["FaceNames"]

# while True:
#     success, frame = cap.read()
#     #img = my_drone.get_frame_read().frame
    
#     original_img = frame.copy()
#     # Detect faces
#     identify_user(frame,FaceNames,FaceEncodings)

#     if cv2.waitKey(1) == ord('q'):
#         break

#     # Display the image with detected faces (optional)
#     cv2.imshow("Image with Detected Faces", original_img)
    

# cap.release() 
# cv2.destroyAllWindows()    
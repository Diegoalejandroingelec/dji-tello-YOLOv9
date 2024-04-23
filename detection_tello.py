#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:25:40 2024

@author: diego
"""


from djitellopy import Tello
import cv2
import threading
import time
import pygame
from datetime import datetime
import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.general import ( Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import operator 
import mediapipe as mp

pygame.init()
def load_model(weights,device,imgsz):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    return model,names,pt

weights = 'v9_c_best.pt'
device = 0 # cuda device, i.e. 0 or 0,1,2,3 or cpu
imgsz=(640, 640)
model,names,_=load_model(weights,device,imgsz)


window = pygame.display.set_mode(imgsz)



mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def draw_face_and_hands(frame, face_landmarks, hand_landmarks):
    # Get the center of the frame (x-coordinate)
    frame_h, frame_w, _ = frame.shape
    frame_center_x = frame_w // 2
    frame_center_y = frame_h // 2
    
    h, w, c = frame.shape
    # Draw the face mesh and bounding box
    face_center_x = frame_center_x
    face_center_y = frame_center_y
    
    if face_landmarks:
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
            # Determine the center of the face
            face_center_x = (x_max + x_min) // 2
            face_center_y = (y_max + y_min) // 2

            # Draw the center point of the face
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
            
            # Draw face bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            

            mp_drawing.draw_landmarks(
                frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
            )

    # Draw the hands landmarks and bounding box
    if hand_landmarks:
        for landmarks in hand_landmarks:
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
            # Draw hand bounding box
            #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # mp_drawing.draw_landmarks(
            #     frame, landmarks, mp_hands.HAND_CONNECTIONS,
            #     mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            #     mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            # )


    return frame, face_center_x, face_center_y






# Global variable to signal upward movement

signal_takeoff=False
signal_up = False
signal_down=False
signal_left=False
signal_right=False
signal_land=False
signal_picture=False
signal_backward=False
signal_forward=False
flying = False
is_rotating_clockwise = False
is_rotating_counter_clockwise = False
cmd = "_"



def inference(im,model,names,line_thickness):

    dt = (Profile(), Profile(), Profile())
    im=np.expand_dims(im,0)
    #im = im[..., ::-1].transpose((0, 3, 1, 2))
    im = im.transpose((0, 3, 1, 2))
    im0s=im.copy()
    im0s=np.squeeze(im0s,0)
    im0s= np.transpose(im0s, (1,2,0))
    with dt[0]:
        im = torch.from_numpy(im.copy()).to(model.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        
        pred = model(im, augment=False, visualize=False)
        pred = pred[0][1]

    # NMS
    with dt[2]:
        pred = non_max_suppression(prediction=pred,
                                    conf_thres=0.50,
                                    iou_thres=0.45,
                                    agnostic=False,
                                    max_det=1000)



    # Process predictions 
    det = pred[0]
    annotator = Annotator(np.ascontiguousarray(im0s), line_width=line_thickness, example=str(names))
    predicted_classes = []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            # Add bbox to image
            c = int(cls)  # integer class
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
            predicted_classes.append((names[c],conf.item()))

    # Stream results
    im0 = annotator.result()
    
    return predicted_classes,im0


def get_frame():
    global signal_takeoff
    global signal_up 
    global signal_down
    global signal_left
    global signal_right
    global signal_land
    global signal_picture
    global signal_backward
    global signal_forward
    global cmd
    global is_rotating_clockwise
    global is_rotating_counter_clockwise
    #cap =  cv2.VideoCapture('udp://' + te_ip + ':11111'+'?overrun_nonfatal=1&fifo_size=50000000')
    
    flag = True
    i=0
   

    
    threshold = 10
    previous_predicted_class = []
    classes_counter = {'left':0,'right':0,'up':0,'down':0,'backward':0,'forward':0,'land':0,'picture':0}
    
    
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
         
        while flag:
            # ret, frame = cap.read()
            
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            frame=my_drone.get_frame_read().frame
            if frame.shape==(720, 960, 3):  
                
                frame = cv2.resize(frame, imgsz)    
                predicted_classes,image_with_predictions=inference(frame,model,names,line_thickness=3)
    
    
    
                image = image_with_predictions.copy()
                face_results = face_mesh.process(image)
                hand_results = hands.process(image)
                # image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                frame, face_center_x, face_center_y = draw_face_and_hands(image, face_results.multi_face_landmarks, hand_results.multi_hand_landmarks)
                # cv2.imshow('Face and Hands Detection', image)
                
                
                # Get the center of the frame (x-coordinate)
                frame_h, frame_w, _ = image.shape
                frame_center_x = frame_w // 2
            
                # Draw the center line of the frame
                cv2.line(image, (frame_center_x, 0), (frame_center_x, frame_h), (255, 0, 0), 2)
                cv2.line(image, (frame_center_x + 120, 0), (frame_center_x + 120, frame_h), (0, 0, 255), 2)
                cv2.line(image, (frame_center_x - 120, 0), (frame_center_x - 120, frame_h), (0, 0, 255), 2)
                
        
    
                # Draw the center point of the face
                cv2.circle(image, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
    
                # Calculate the x distance from the face center to the frame's center line
                x_distance = face_center_x - frame_center_x
                #distance_text = f"X Distance: {x_distance}"
                #cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                
                if abs(x_distance) >= 120:
                    direction = "clockwise" if x_distance > 0 else "anti-clockwise"
                    if flying:
                        
                        if direction == "clockwise":
                            cmd = "Rotating Clockwise"
                            print("Rotating Clockwise")
                            is_rotating_clockwise = True
                        else:
                            cmd = "Rotating Counter-Clockwise"
                            print("Rotating Counter Clockwise")
                            is_rotating_counter_clockwise = True

                                       
                    # distance2move = f"X Distance to move: {abs(120 - abs(x_distance))} {direction}"
                    # cv2.putText(frame, distance2move, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
                # Visualize the x distance
                cv2.line(image, (face_center_x, face_center_y), (frame_center_x, face_center_y), (250, 255, 0), 2)

    
    
    
                # image_with_faces, detected_faces = detect_faces(frame.copy())
                #frame_rgb = image_with_predictions
                battery = my_drone.get_battery()
                image = cv2.putText(image, f'Battery: {str(battery)} %', (frame_center_x, 600), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 255, 255), 1, cv2.LINE_AA)
                image = cv2.putText(image, f'Command: {cmd}', (frame_center_x, 570), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 255, 255), 1, cv2.LINE_AA)
                
                
                
                
                
                frame_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
                
                # Blit and display
                window.blit(frame_surface, (0, 0))
                pygame.display.update()
                
                if(predicted_classes):
                    
                    highest_prob_class = max(predicted_classes, key=operator.itemgetter(1))
                    classes_counter[highest_prob_class[0]]+=1
    
                    if(previous_predicted_class and  highest_prob_class[0]!=previous_predicted_class):
                        classes_counter[previous_predicted_class]=0
    
                    exceeding_threshold = {key: value for key, value in classes_counter.items() if value >= threshold}
                    is_empthy=not exceeding_threshold
                    if not is_empthy:
                        print(f'predicted class is: {exceeding_threshold}')
                        classes_counter = {'left':0,'right':0,'up':0,'down':0,'backward':0,'forward':0,'land':0,'picture':0}
                        class_action=list(exceeding_threshold.keys())[0]
                        if class_action == 'up':
                            signal_up = True
                            cmd = "Move UP"
                        elif class_action == 'left':
                            signal_right = True
                            cmd = "Move LEFT"
                        elif class_action == 'down':
                            signal_down = True
                            cmd = "Move DOWN"
                        elif class_action == 'right':
                            signal_left = True
                            cmd = "Move RIGHT"
                        elif class_action == 'picture':
                            frame_RGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(f'./pictures/image_{i}_{current_time}.jpg', frame_RGB)
                            i += 1
                            signal_picture = True
                            cmd = "Take PICTURE"
                        elif class_action == 'land':
                            signal_land = True
                            cmd = "LAND"
                        elif class_action == 'backward':
                            signal_backward=True
                            cmd = "BACK"
                        elif class_action == 'forward':
                            signal_forward=True
                            cmd = "FORWARD"
                            
                        class_action=''
                            
                    previous_predicted_class=highest_prob_class[0] 
                    
    
                    
                    
                        
                # Handle Pygame events
                for event in pygame.event.get():
                     if event.type == pygame.KEYDOWN:
                         if event.key == pygame.K_q:
                             flag = False
                         elif event.key == pygame.K_w:
                             signal_up = True
                             cmd = "Move UP"
                         elif event.key == pygame.K_a:
                             signal_left = True
                             cmd = "Move LEFT"
                         elif event.key == pygame.K_s:
                             signal_down = True
                             cmd = "Move DOWN"
                         elif event.key == pygame.K_d:
                             signal_right = True
                             cmd = "Move RIGHT"
                         elif event.key == pygame.K_p:
                             frame_RGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                             cv2.imwrite(f'./pictures/image_{i}_{current_time}.jpg', frame_RGB)
                             i += 1
                             signal_picture = True
                             cmd = "Take PICTURE"
                         elif event.key == pygame.K_t:
                             signal_takeoff = True
                             cmd = "TAKE-OFF"
                         elif event.key == pygame.K_l:
                             signal_land = True
                             cmd = "LAND"
                         elif event.key == pygame.K_b:
                             signal_backward=True
                             cmd = "BACK"
                         elif event.key == pygame.K_f:
                             signal_forward=True
                             cmd = "FORWARD"
                     elif event.type == pygame.QUIT:
                         flag = False

    
    pygame.quit()

def control_drone():

    global signal_takeoff
    global signal_up 
    global signal_down
    global signal_left
    global signal_right
    global signal_land
    global signal_backward
    global signal_forward
    global cmd
    global flying
    global is_rotating_clockwise
    global is_rotating_counter_clockwise
    
    while True:
        if signal_takeoff:
            print('Drone take off')
            my_drone.takeoff()
            signal_takeoff = False
            cmd = "_"
            flying = True
        if signal_up:
            print('Drone up')
            my_drone.move_up(20)
            signal_up = False
            cmd = "_"
        if signal_down:
            print('Drone down')
            my_drone.move_down(20)
            signal_down = False
            cmd = "_"
        if signal_left:
            print('Drone left')
            my_drone.move_left(20)
            signal_left = False
            cmd = "_"
        if signal_right:
            print('Drone right')
            my_drone.move_right(20)
            signal_right = False
            cmd = "_"
        if signal_backward:
            print('Drone back')
            my_drone.move_back(20)
            signal_backward = False
            cmd = "_"
        if signal_forward:
            print('Drone forward')
            my_drone.move_forward(20)
            signal_forward = False
            cmd = "_"
        if signal_land:
            print('Drone land')
            my_drone.land()
            signal_land = False
            cmd = "_"
        if is_rotating_clockwise:
            print('Drone is rotating clock-wise')
            my_drone.rotate_clockwise(20)
            is_rotating_clockwise = False
            cmd = "_"

            
        if is_rotating_counter_clockwise:
            print('Drone is rotating clock-wise')
            my_drone.rotate_counter_clockwise(20)
            is_rotating_counter_clockwise= False
            cmd = "_"
            
        time.sleep(0.1)  # Add a short delay to prevent this loop from consuming too much CPU

my_drone = Tello()
my_drone.connect()
 
my_drone.streamon()





#Start the get_frame thread
get_frame_thread = threading.Thread(target=get_frame)
get_frame_thread.daemon = True
get_frame_thread.start()

#Start the control_drone thread
control_drone_thread = threading.Thread(target=control_drone)
control_drone_thread.daemon = True
control_drone_thread.start()

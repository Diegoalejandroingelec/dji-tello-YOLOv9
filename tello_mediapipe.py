from djitellopy import Tello
import cv2
import threading
import time
import pygame
from datetime import datetime
import mediapipe as mp
import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.general import ( Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import operator
from FaceRecognitionC import identify_user
import pickle
pygame.init()


NUMBERS ={
          '0':'rr0000rrr0rrrr0r0rrrrrr00r0rr0r0rrrrrrrrr0rrrr0rrr0000rr0rrrrrr0',
          '1':"0000r00000rrr00000r0r0000000r0000000r0000000r0000000r0000rrrrrr0",
          '2':'0rrrrr000r000r0000000r000000r000000r000000r000000r000r000rrrrr00',
          '3':'0rrrrr000r000r0000000r000000r0000000r00000000r000r000r000rrrrr00'}

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

screen_width = 960
screen_height = 720
window = pygame.display.set_mode((screen_width, screen_height))


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

with open('core_embeddings.pkl', 'rb') as f:
# with open('embeddings.pkl', 'rb') as f:
    FaceEmbeddings = loaded_embeddings_dict = pickle.load(f)

FaceEncodings= FaceEmbeddings["FaceEmbeddings_new"]
FaceNames = FaceEmbeddings["FaceNames"]

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
is_rotating_clockwise = False
is_rotating_counter_clockwise = False
cmd = "_"
IMAGE_DIR = './assets/'
original_frame = None
start_time = time.time()
person = '-'
is_authenticated = False
picture_counter = -1
countdown_started_time = None

flashing = False
# Create a surface for the flashing effect
flash_surface = pygame.Surface((screen_width, screen_height))
flash_surface.fill((255, 255, 255))  # White flash

# Set up flashing parameters
flash_duration = 1.0  # Total duration of the flash effect in seconds
flash_time= 0

def draw_outlined_text(image, text, position, font, font_scale, color, thickness, outline_color, outline_thickness):
    image = cv2.putText(image, text, position, font, font_scale, outline_color, outline_thickness, lineType=cv2.LINE_AA)
    image = cv2.putText(image, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return image


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
                                    conf_thres=0.70,
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

def perform_face_recognition(original_frame):
    global start_time
    global person
    global is_authenticated
    try:
        current_time = time.time()  # Get the current time
        
        elapsed_time = current_time - start_time  # Calculate elapsed time
        # print(elapsed_time)
        if elapsed_time >= 10:  # Check if 10 seconds have passed
            # print('AUTHENTICATION IN PROGRESS--------------------------------------------------------')
            is_authenticated, person = identify_user(original_frame,FaceNames,FaceEncodings)
            start_time = time.time()  # Reset the start time
    except:
        pass
    
def picture_countdown_completed():
    global picture_counter
    global countdown_started_time
    if picture_counter == -1 or countdown_started_time is None:
        return False, False # Not finished, there is no countdown
    current_time = time.time()
    if current_time - countdown_started_time >= 1:
        countdown_started_time = time.time()
        picture_counter -= 1
        number_to_show=NUMBERS[str(picture_counter)]
        
        my_drone.send_expansion_command(f"mled sg {number_to_show}")
        if picture_counter <= 0:
            return True, True # Finished, there is a countdown
        return False, True # Not finished, there is a countdown
    
    return False, False # Not planned condition
    
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
    global original_frame
    global person
    global is_authenticated
    global picture_counter
    global countdown_started_time
    global flashing
    # Create a surface for the flashing effect
    global flash_surface
    # Set up flashing parameters
    global flash_duration 
    global half_flash_duration
    global time_elapsed
    
    
    flying = False
    # cap =  cv2.VideoCapture(0)
    # cap =  cv2.VideoCapture('udp://' + te_ip + ':11111'+'?overrun_nonfatal=1&fifo_size=50000000')
    flag = True
    i=0
    
    threshold = 10
    previous_predicted_class = []
    classes_counter = {'left':0,'right':0,'up':0,'down':0,'backward':0,'forward':0,'land':0,'picture':0}
    
    ctr = 0
    
    # Text attributes for buttons
    button_images = [pygame.image.load(f'{IMAGE_DIR}up.png').convert_alpha(), 
                     pygame.image.load(f'{IMAGE_DIR}right.png').convert_alpha(), 
                     pygame.image.load(f'{IMAGE_DIR}down.png').convert_alpha(), 
                     pygame.image.load(f'{IMAGE_DIR}left.png').convert_alpha(), 
                     pygame.image.load(f'{IMAGE_DIR}picture.png').convert_alpha(), 
                     pygame.image.load(f'{IMAGE_DIR}backward.png').convert_alpha(), 
                     pygame.image.load(f'{IMAGE_DIR}take-off.png').convert_alpha(),
                     pygame.image.load(f'{IMAGE_DIR}land.png').convert_alpha(), 
                     pygame.image.load(f'{IMAGE_DIR}forward.png').convert_alpha(),
                     ]
    frame_image = pygame.image.load(f'{IMAGE_DIR}frame.png')
    frame_image = pygame.transform.scale(frame_image, (screen_width, screen_height))
    button_rects = []
    original_buttons = []
    button_positions = [(800,500), (865, 565), (800, 630), (735, 565), (800, 565), (350,640), (415,640), (480, 640), (545, 640)]
    frame_position = (0,0)
    
    for i in range(len(button_images)):
        button_images[i] = pygame.transform.scale(button_images[i], (65, 65))
        button_rects.append(button_images[i].get_rect(topleft=(button_positions[i])))
        original_buttons.append(button_images[i].copy())

    # Alpha values for button transparency
    default_alpha = 255  # Fully opaque
    hover_alpha = 100  # Semi-transparent
    click_alpha = 50   # More transparent on click
    
    click_effect_duration = 200  # Duration of the click effect in milliseconds
    last_click_time = 0  # Timestamp of the last click
    
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
        while flag:
            # ret, frame = cap.read()
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            frame=my_drone.get_frame_read().frame
            original_frame = frame.copy() 
            
            if frame.shape==(720, 960, 3):  

                frame = cv2.resize(frame, imgsz)    
                predicted_classes,image_with_predictions=inference(frame,model,names,line_thickness=3)
                
                # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = image_with_predictions.copy()
                face_results = face_mesh.process(image)
                hand_results = hands.process(image)
                # image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                frame, face_center_x, face_center_y = draw_face_and_hands(image, face_results.multi_face_landmarks, hand_results.multi_hand_landmarks)
                # cv2.imshow('Face and Hands Detection', image)
                # frame_rgb = image
                
                # Get the center of the frame (x-coordinate)
                frame_h, frame_w, _ = image.shape
                frame_center_x = frame_w // 2
            
                # Draw the center line of the frame
                # cv2.line(image, (frame_center_x, 0), (frame_center_x, frame_h), (255, 0, 0), 2)
                # cv2.line(image, (frame_center_x + 120, 0), (frame_center_x + 120, frame_h), (0, 0, 255), 2)
                # cv2.line(image, (frame_center_x - 120, 0), (frame_center_x - 120, frame_h), (0, 0, 255), 2)
    
                # Draw the center point of the face
                # cv2.circle(image, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
    
                # Calculate the x distance from the face center to the frame's center line
                x_distance = face_center_x - frame_center_x
                #distance_text = f"X Distance: {x_distance}"
                #cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if abs(x_distance) >= 120:
                    direction = "clockwise" if x_distance > 0 else "anti-clockwise"
                    if flying:
                        
                        if direction == "clockwise":
                            cmd = "Rotating Clockwise"
                            is_rotating_clockwise = True
                        else:
                            cmd = "Rotating Counter-Clockwise"
                            is_rotating_counter_clockwise = True

                                       
                    # distance2move = f"X Distance to move: {abs(120 - abs(x_distance))} {direction}"
                    # cv2.putText(frame, distance2move, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
                # Visualize the x distance
                # cv2.line(image, (face_center_x, face_center_y), (frame_center_x, face_center_y), (250, 255, 0), 2)
                
                
                battery = my_drone.get_battery()
                # battery = 100
                
                image=cv2.resize(image,(960,720))
                image = draw_outlined_text(image, f'Battery: {str(battery)} %', (760, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, (255, 255, 255), 5)
                image = draw_outlined_text(image, f'Command: {cmd}', (30, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, (255, 255, 255), 5)
                image = draw_outlined_text(image, f'Person: {person}', (30, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, (255, 255, 255), 5)
                
                countdown_finished, countdown = picture_countdown_completed()
                if countdown and picture_counter >= 0:
                    print(countdown, countdown_finished)
                    image = draw_outlined_text(image, f'{int(picture_counter)}', (int(screen_width/2)-45, int(screen_height/2)+35), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 6, (255, 255, 255), 10)
                    if countdown_finished:
                        picture_counter = -1
                        countdown_started_time = None
                        flashing = countdown_finished
                        flash_time = time.time()
                        
                        frame_RGB=cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(f'./pictures/image_{i}_{current_time}.jpg', frame_RGB)
                        
                frame_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
                
                
                # Blit and display
                window.blit(frame_surface, (0, 0))
                window.blit(frame_image, (0, 0))
                
                if flashing:
                    current_time = time.time()
                    if current_time - flash_time <= flash_duration:  
                        window.blit(flash_surface, (0, 0))
                    else:
                        flashing = False
                
                current_time_pygame = pygame.time.get_ticks()
                
                # Check for mouse hover
                mouse_pos = pygame.mouse.get_pos()
                
                for i in range(len(button_rects)):
                    if button_rects[i].collidepoint(mouse_pos):
                        if current_time_pygame - last_click_time > click_effect_duration:
                            # Change the alpha value when hovering
                            button_images[i].set_alpha(hover_alpha)
                    else:
                        # Reset the alpha value when not hovering or clicking
                        if current_time_pygame - last_click_time > click_effect_duration:
                            button_images[i] = original_buttons[i].copy()
                            button_images[i].set_alpha(default_alpha)
                            
                    window.blit(button_images[i], button_rects[i])
    
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
                            picture_counter = 4
                            countdown_started_time = time.time()
                            print("TAKING PICTURE")
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
                             picture_counter = 4
                             countdown_started_time = time.time()
                             print("TAKING PICTURE")
                             i += 1
                             signal_picture = True
                             cmd = "Take PICTURE"
                         elif event.key == pygame.K_t:
                             signal_takeoff = True
                             cmd = "TAKE-OFF"
                             flying = True
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
                         
                     # Mouse button down event
                     elif event.type == pygame.MOUSEBUTTONDOWN:
                         for i in range(len(button_rects)):
                             if button_rects[i].collidepoint(event.pos):
                                 button_images[i].set_alpha(click_alpha)
                                 last_click_time = current_time_pygame
                                 if i == 0:
                                     signal_up = True
                                     cmd = "Move UP"
                                 elif i == 3:
                                     signal_left = True
                                     cmd = "Move LEFT"
                                 elif i == 2:
                                     signal_down = True
                                     cmd = "Move DOWN"
                                 elif i == 1:
                                     signal_right = True
                                     cmd = "Move RIGHT"
                                 elif i == 4:
                                     picture_counter = 4
                                     countdown_started_time = time.time()
                                     i += 1
                                     signal_picture = True
                                     cmd = "Take PICTURE"
                                     print("TAKING PICTURE")
                                 elif i == 6:
                                     signal_takeoff = True
                                     cmd = "TAKE-OFF"
                                     flying = True
                                 elif i == 7:
                                     signal_land = True
                                     cmd = "LAND"
                                 elif i == 5:
                                     signal_backward=True
                                     cmd = "BACK"
                                 elif i == 8:
                                     signal_forward=True
                                     cmd = "FORWARD"
                    
                    
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
    global is_rotating_clockwise
    global is_rotating_counter_clockwise
    global original_frame
    global start_time
    global is_authenticated
    global person
    global signal_picture
    #my_drone.send_expansion_command("mled sc")
    
    command_1_sent=False
    command_2_sent=False
    while True:
        perform_face_recognition(original_frame)
        if(is_authenticated):
            
            if(not command_1_sent):
                my_drone.send_expansion_command("led bl 0.5 0 255 0 0 255 0")
                my_drone.send_expansion_command("mled sg 00rrrr000r0000r0r0r00r0rr000000rr0r00r0rr00rr00r0r0000r000rrrr00")
                command_2_sent = False
                command_1_sent = True
                
            if signal_picture:
                print('Drone led')
                #my_drone.send_expansion_command("mled l r 1.5 PICTURE")
                
            
                #my_drone.send_expansion_command("mled sg 00rrrr000r0000r0r0r00r0rr000000rr0r00r0rr00rr00r0r0000r000rrrr00")
               
                
                my_drone.send_expansion_command("led bl 0.5 255 0 255 0 0 255")
                
                #my_drone.send_expansion_command("mled sc") 
                signal_picture = False
                cmd = "_"
            if signal_takeoff:
                print('Drone take off')
                led_take_off = "00000000000rr00000r00r000r0000r0r00rr00r00r00r000r0000r0r000000r"
                # my_drone.send_expansion_command("mled l g 1.5 GESTURE FLY")
                my_drone.send_expansion_command(f"mled u g 1.5 {led_take_off}")
                my_drone.takeoff()
                signal_takeoff = False
                cmd = "_"
            if signal_up:
                print('Drone up')
                led_up = "000r000000rrr0000r0r0r00r00r00r0000r0000000r0000000r000000000000"
                # my_drone.send_expansion_command("mled u r 1.5 UP")
                my_drone.send_expansion_command(f"mled u g 1.5 {led_up}")
                my_drone.move_up(20)
                signal_up = False
                cmd = "_"
            if signal_down:
                print('Drone down')
                led_down = "r0000000000r0000000r0000000r0000r00r00r00r0r0r0000rrr000000r0000"
                # my_drone.send_expansion_command("mled d r 1.5 DOWN")
                my_drone.send_expansion_command(f"mled d g 1.5 {led_down}")
                my_drone.move_down(20)
                signal_down = False
                cmd = "_"
            if signal_left:
                print('Drone left')
                led_right = "000000000000r00000000r00000000r00rrrrrrr000000r000000r000000r000"
                # my_drone.send_expansion_command("mled r r 1.5 RIGHT")
                my_drone.send_expansion_command(f"mled r g 1.5 {led_right}")
                my_drone.move_left(20)
                signal_left = False
                cmd = "_"
            if signal_right:
                print('Drone right')
                led_left = "00000000000r000000r000000r000000rrrrrrr00r00000000r00000000r0000"
                # my_drone.send_expansion_command("mled l r 1.5 LEFT")
                my_drone.send_expansion_command(f"mled l g 1.5 {led_left}")
                my_drone.move_right(20)
                signal_right = False
                cmd = "_"
            if signal_backward:
                print('Drone back')
                led_back = "rrr00000r0r00r000r0000r0rrr0000r0r0r00r00r000r00r0r00000r0rr0000"
                # my_drone.send_expansion_command("mled l r 1.5 BACK")
                my_drone.send_expansion_command(f"mled sg {led_back}")
                my_drone.move_back(20)
                signal_backward = False
                cmd = "_"
            if signal_forward:
                print('Drone forward')
                led_forward = "rrr00000r0r0000r0r0000r0rrr00r000r0r00r00r00000rr0r00000r0rr0000"
                # my_drone.send_expansion_command("mled l r 1.5 FORWARD")
                my_drone.send_expansion_command(f"mled sg {led_forward}")
                my_drone.move_forward(20)
                signal_forward = False
                cmd = "_"
            if signal_land:
                print('Drone land')
                led_land = "r000000r0r0000r000r00r00000rr000r000000r0r0000r000r00r00000rr000"
                # my_drone.send_expansion_command("mled l r 1.5 LANDING")
                my_drone.send_expansion_command(f"mled sg {led_land}")
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
        else:
            if(not command_2_sent):
                my_drone.send_expansion_command("led bl 0.5 255 0 0 254 0 0")
                my_drone.send_expansion_command("mled sg 00rrrr000r0000r0r0r00r0rr000000rr00rr00rr0r00r0r0r0000r000rrrr00")
                command_2_sent = True
                command_1_sent = False
                
            person = 'NOT AUTHENTICATED'
            cmd = "_" 
            
        time.sleep(0.1)  # Add a short delay to prevent this loop from consuming too much CPU

my_drone = Tello()
my_drone.connect()
 
my_drone.streamon()


# Start the get_frame thread
get_frame_thread = threading.Thread(target=get_frame)
get_frame_thread.daemon = True
get_frame_thread.start()

# Start the control_drone thread
control_drone_thread = threading.Thread(target=control_drone)
control_drone_thread.daemon = True
control_drone_thread.start()
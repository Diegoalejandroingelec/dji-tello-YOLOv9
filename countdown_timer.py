import cv2
import time

def countdown_timer(frame, count):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(count)
    # Get dimensions of the text
    text_size = cv2.getTextSize(text, font, 4, 2)[0]
    # Calculate X, Y coordinates of the text
    text_x = int((frame.shape[1] - text_size[0]) / 2)
    text_y = int((frame.shape[0] + text_size[1]) / 2)
    # Put text on the frame
    cv2.putText(frame, text, (text_x, text_y), font, 4, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def take_picture():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            for count in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break
                frame = countdown_timer(frame, count)
                cv2.imshow('frame', frame)
                cv2.waitKey(1000)  # Wait for 1 second
            ret, frame = cap.read()  # Capture the frame after the countdown
            cv2.imshow('Captured Image', frame)
            cv2.imwrite('captured_image.png', frame)  # Save the captured image
            cv2.waitKey(2000)  # Display the captured image for 2 seconds

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    take_picture()

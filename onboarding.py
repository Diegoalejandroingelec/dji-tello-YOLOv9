import pygame
import sys
import os
import tkinter as tk
from tkinter import filedialog
import shutil
from tkinter import messagebox
import subprocess

uploaded = False

# Function to handle file uploads
def upload_image():
    global uploaded
    global input_name
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open the file dialog
    if file_path:
        print(f"File selected: {file_path}")
        try:
            # Assuming you want to copy the selected file to the current directory
            destination = os.path.join(os.getcwd(), input_name+'.png')
            shutil.copy(file_path, destination)
            messagebox.showinfo("Success", "File uploaded successfully!")
            uploaded = True
        except Exception as e:
            print(f"Error: {e}")

# Initialize Pygame
pygame.init()

# Set the screen to full size
infoObject = pygame.display.Info()
screen_width, screen_height = infoObject.current_w, infoObject.current_h
screen = pygame.display.set_mode((screen_width, screen_height))

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (100, 100, 100)

# Fonts
font = pygame.font.SysFont("Arial", 24)
font_title = pygame.font.SysFont("Arial", 30, bold=True)
font_button = pygame.font.SysFont("Arial", 24, bold=True)

# Load images from the assets directory
drone_icon = pygame.image.load('assets/drone.png')
wifi_icon = pygame.image.load('assets/wifi.png')
lock_icon = pygame.image.load('assets/photo.png')
logo = pygame.image.load('assets/logo.png')

# Scale images if necessary
size = 96
drone_icon = pygame.transform.scale(drone_icon, (size, size))
wifi_icon = pygame.transform.scale(wifi_icon, (size, size))
lock_icon = pygame.transform.scale(lock_icon, (size, size))
logo = pygame.transform.scale(logo, (size * 2, size * 2))  # Scaled for visibility

# Logo and Title
logo_pos = (screen_width // 2 - logo.get_width() // 2, 20)
title_text = font_title.render('Welcome to GestureFly', True, BLACK)
title_pos = (screen_width // 2 - title_text.get_width() // 2, logo_pos[1] + logo.get_height() + 20)

# Instructions and their icons
instructions = [
    (drone_icon, 'Get your drone ready in a wide space'),
    (wifi_icon, 'Connect your device to TELLO-XXXX WiFi connection'),
    (lock_icon, "Upload your face photo for drone's authentication")
]

# Calculate total height of instructions to adjust the starting y position
space_between_instructions = 10
total_instructions_height = sum([icon.get_height() for icon, _ in instructions]) + (len(instructions) - 1) * space_between_instructions
start_y = title_pos[1] + title_text.get_height() + 40  # Additional margin

# Calculate vertical space between the instructions and the input area
margin_between_instructions_and_input = 70  # Additional margin

# Input box and buttons positions
input_box_y = start_y + total_instructions_height + margin_between_instructions_and_input
input_box = pygame.Rect(screen_width // 2 - 150, input_box_y, 300, 50)
button_y = input_box.bottom + 30  # Additional margin
upload_button = pygame.Rect(screen_width // 2 - 150 - 10, button_y, 150, 50)
action_button = pygame.Rect(screen_width // 2 + 10, button_y, 150, 50)

# Input name
input_name = ""
base_font = pygame.font.Font(None, 40)
active = False
color_inactive = pygame.Color('lightskyblue3')
color_active = pygame.Color('dodgerblue2')
color = color_inactive

# Placeholder text
placeholder_text = "Input your name.."
input_name = placeholder_text

# Main game loop
running = True
while running:
    screen.fill(WHITE)
    mouse_pos = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        
        # Event handling for input box and buttons
        if event.type == pygame.MOUSEBUTTONDOWN:
            if input_box.collidepoint(mouse_pos):
                active = not active
                if active and input_name == placeholder_text:
                    # Clear placeholder when clicked
                    input_name = ""
            else:
                active = False
                if not input_name:
                    # Restore placeholder when clicking outside the box
                    input_name = placeholder_text

            if upload_button.collidepoint(mouse_pos):
                # Add functionality for upload button click
                if input_name and input_name != placeholder_text:
                    upload_image()
                else:
                    messagebox.showinfo("Failed", "Please fill the name first!")
            if action_button.collidepoint(mouse_pos):
                # Add functionality for action button click
                if uploaded:
                    pygame.quit()
                    running = False
                    subprocess.run(["python", "tello_mediapipe.py"], check=True)
                else:
                    messagebox.showinfo("Failed", "Please upload your face photo!")

        if event.type == pygame.KEYDOWN:
            if active:
                if event.key == pygame.K_RETURN:
                    # Add functionality for return key
                    print(input_name)
                elif event.key == pygame.K_BACKSPACE:
                    input_name = input_name[:-1]
                else:
                    input_name += event.unicode

    # Input box text
    txt_surface = base_font.render(input_name, True, BLACK)
    width = max(300, txt_surface.get_width() + 10)
    input_box.w = width
    screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
    pygame.draw.rect(screen, color, input_box, 2)

    # Draw logo and title
    screen.blit(logo, logo_pos)
    screen.blit(title_text, title_pos)

    # Draw instructions
    for i, (icon, text) in enumerate(instructions):
        icon_x = screen_width // 2 - (icon.get_width() + font.size(text)[0] + space_between_instructions) // 2
        text_x = icon_x + icon.get_width() + space_between_instructions
        y = start_y + i * (icon.get_height() + space_between_instructions)
        screen.blit(icon, (icon_x, y))
        screen.blit(font.render(text, True, BLACK), (text_x, y + icon.get_height() // 2 - font.size(text)[1] // 2))

    # Draw buttons with hover effects
    upload_button_color = LIGHT_GRAY if upload_button.collidepoint(mouse_pos) else GRAY
    action_button_color = LIGHT_GRAY if action_button.collidepoint(mouse_pos) else GRAY
    pygame.draw.rect(screen, upload_button_color, upload_button)
    pygame.draw.rect(screen, action_button_color, action_button)
    screen.blit(font_button.render('Upload', True, BLACK), (upload_button.x + (upload_button.width - font_button.size('Upload')[0]) // 2, upload_button.y + (upload_button.height - font_button.size('Upload')[1]) // 2))
    screen.blit(font_button.render('Action', True, BLACK), (action_button.x + (action_button.width - font_button.size('Action')[0]) // 2, action_button.y + (action_button.height - font_button.size('Action')[1]) // 2))

    pygame.display.flip()

pygame.quit()
sys.exit()

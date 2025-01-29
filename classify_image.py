import sys
sys.path.insert(0, "/home/pi/myenv/lib/python3.11/site-packages")

import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import lgpio as GPIO
import time

# GPIO Pin for Servo Motor and PIR Sensor
SERVO_PIN = 18
PIR_PIN = 17  
CHIP = 0

# Servo motor control logic
def move_servo(position):
    duty_cycle = 2.5 + (position / 180.0) * 10.0 
    GPIO.tx_pwm(h, SERVO_PIN, 50, duty_cycle)  
    print(f"Servo moved to {position} degrees.")
    time.sleep(1) 

def reset_servo():
    move_servo(90) 

# Model Loading and Preprocessing
def load_model(model_path, device):
    model = models.resnet50(weights=None) 
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(num_features, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

def classify_image(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

def capture_image(output_path="captured_image.jpg"):
    command = f"libcamera-still -o {output_path} --width 1640 --height 1232 --timeout 1000"
    os.system(command)
    print(f"Image saved as {output_path}")

# Motion Detection and Classification
def motion_detected(model, output_path, class_names, device):
    print("Motion detected! Waiting 3 seconds before capturing...")
    time.sleep(3)  
    capture_image(output_path)

    print("Classifying image...")
    image_tensor = preprocess_image(output_path).to(device)
    prediction = classify_image(model, image_tensor, class_names)
    print(f"Predicted Class: {prediction}")

    # Move the servo based on the prediction
    if prediction == "plastic_bottle":
        print("Sorting as plastic bottle.")
        move_servo(0) 
    else:
        print("Sorting as other waste.")
        move_servo(180)

    time.sleep(2) 
    reset_servo() 

# Main Function
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Path to model and temporary image file
    model_path = "resnet_model.pth"
    output_path = "captured_image.jpg"
    class_names = ["others", "plastic_bottle"]

    print("Loading model...")
    model = load_model(model_path, device)

    # Initialize GPIO
    h = GPIO.gpiochip_open(CHIP)
    GPIO.gpio_claim_input(h, PIR_PIN) 
    GPIO.gpio_claim_output(h, SERVO_PIN)
    reset_servo() 

    print("Waiting for motion...")
    try:
        while True:
            if GPIO.gpio_read(h, PIR_PIN):
                motion_detected(model, output_path, class_names, device)
                time.sleep(2)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        reset_servo()  
        GPIO.gpiochip_close(h)
        print("GPIO cleanup done.")

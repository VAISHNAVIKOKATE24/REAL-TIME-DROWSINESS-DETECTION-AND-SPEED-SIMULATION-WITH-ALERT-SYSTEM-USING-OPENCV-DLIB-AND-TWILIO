# Import required libraries
import pygame # Used to play buzzer sound
import time 
import cv2 # For video processing
import numpy as np
import dlib # Face detection and landmark prediction
from imutils import face_utils # Converts dlib landmarks to numpy
import requests # To fetch IP-based location
import urllib.parse # To format location for a Google Maps URL
from twilio.rest import Client # To send SMS alerts via Twilio
import csv # For logging data
import os # For logging data

# Initialize pygame mixer for buzzer sound
pygame.mixer.init()
pygame.mixer.music.load("beep.mp3")  # Load beep sound

# Use external USB camera (index 1)
cap = cv2.VideoCapture(0)

# Initialize Dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Twilio configuration for SMS alerts
TWILIO_PHONE_NUMBER = "+12316748865"
TO_PHONE_NUMBERS = ["+917666536559", "+919356758210"]
TWILIO_ACCOUNT_SID = "AC26148464020ac07424f6907a5c55b055"
TWILIO_AUTH_TOKEN = "a192ecb3281d7de04bae5920f84bf11a"
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Setup CSV files for logs and events
log_csv = "drowsiness_log.csv"
event_csv = "drowsiness_events.csv"
csv_headers = ["timestamp", "status", "speed_kmh", "location"]

# Create CSVs if they do not exist
for csv_file in [log_csv, event_csv]:
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

# Initialize counters and speed variables
# Tracks eye status and defines speed reduction logic.
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
last_sms_time = 0
simulated_speed = 50  # Initial speed
speed_decrement = 5   # Speed drop on sleep
min_speed = 10
sms_interval = 10     # Time gap between SMS alerts in seconds

# Function to calculate Euclidean distance between two points
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Function to detect eye blink ratio
# Measures eye aspect ratio (EAR) to classify eye state:
# Open (2), Drowsy (1), or Closed (0).

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e) # short distance points
    down = compute(a, f) # long distance points
    ratio = up / (2.0 * down) # divide by long distance points


    if ratio > 0.25: # determined ratio of open eye status
        return 2  # Eyes open
    elif 0.21 < ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Eyes closed

# Function to fetch approximate location using IP
# Uses ipinfo.io API to get approximate latitude and longitude.


def get_ip_location():
    try:
        response = requests.get("http://ipinfo.io/json")
        if response.status_code == 200:
            loc = response.json()['loc'].split(',')
            return loc[0].strip(), loc[1].strip()
    except Exception as e:
        print(f"❌ Location error: {e}")
    return "Unknown", "Unknown"

# Function to send SMS using Twilio
def send_sms(body):
    success = False
    for number in TO_PHONE_NUMBERS:
        try:
            message = client.messages.create(
                body=body,
                from_=TWILIO_PHONE_NUMBER,
                to=number
            )
            print(f"📤 SMS sent to {number}")
            success = True
        except Exception as e:
            print(f"❌ SMS error to {number}: {e}")
    return success

# Function to log data to CSV files
# Logs data to log_csv always, and to event_csv if it's a critical state (drowsy or sleeping).

def log_to_csv(timestamp, status, speed, location, is_event=False):
    try:
        with open(log_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, status, speed, location])
        if is_event:
            with open(event_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, status, speed, location])
        print("📝 Logged to CSV")
    except Exception as e:
        print(f"❌ Logging error: {e}")

# Get initial location using IP
lat, lon = get_ip_location()
print(f"📍 Initial Location: Latitude {lat}, Longitude {lon}")

# Main loop for real-time video processing
# Reads frames from webcam continuously.
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame grab failed")
        continue

# Converts to grayscale and detects faces.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_frame = None

    for face in faces:
        # Draw rectangle around face
        # Draws rectangle and extracts face area.

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_frame = frame[y1:y2, x1:x2].copy()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Detect landmarks
        landmarks = predictor(gray, face)
        # Converts facial landmarks to array.
        landmarks = face_utils.shape_to_np(landmarks)

        # Check for eye blink (left and right)
        # Calculates blink ratio for both eyes.
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                             landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44],
                              landmarks[47], landmarks[46], landmarks[45])

        # Create map link from location
        query = urllib.parse.quote_plus(f"{lat},{lon}")
        map_link = f"https://www.google.com/maps?q={query}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Sleep detection
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                if status != "SLEEPING !!!":
                    status = "SLEEPING !!!"
                    color = (0, 0, 255)

                simulated_speed = max(min_speed, simulated_speed - speed_decrement)

                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
                    print("🔔 Alarm Playing...")

                alert_msg = (
                    f"🚨 *ALERT!* 🚨\n"
                    f"😴 Status: {status}\n"
                    f"🚗 Speed: {simulated_speed} km/h\n"
                    f"📍 Location: {map_link}\n"
                    f"⚠️ Possible accident!\n"
                )

                if time.time() - last_sms_time > sms_interval:
                    print(f"\n😴 {status} | Speed: {simulated_speed}")
                    if send_sms(alert_msg):
                        last_sms_time = time.time()

                log_to_csv(timestamp, status, simulated_speed, map_link, is_event=True)

        # Drowsiness detection
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                if status != "Drowsy !":
                    status = "Drowsy !"
                    color = (0, 0, 200)
                    simulated_speed = 40

                alert_msg = (
                    f"😵 *Drowsiness Detected!* 😵\n"
                    f"Status: {status}\n"
                    f"🚗 Speed: {simulated_speed} km/h\n"
                    f"📍 Location: {map_link}\n"
                    f"⚠️ Stay Alert!\n"
                )

                if time.time() - last_sms_time > sms_interval:
                    print(f"\n👁️ {status} | Speed: {simulated_speed}")
                    if send_sms(alert_msg):
                        last_sms_time = time.time()

                log_to_csv(timestamp, status, simulated_speed, map_link, is_event=True)

        # Active state
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                if status != "Active :)":
                    status = "Active :)"
                    color = (0, 128, 0)
                    simulated_speed = 50
                    print(f"\n✅ {status} | Speed: {simulated_speed}")
                    log_to_csv(timestamp, status, simulated_speed, map_link)

        # Draw landmarks on the face
        # Draws facial landmarks on a separate window.
        if face_frame is not None and face_frame.size != 0:
            for (x, y) in landmarks:
                try:
                    cv2.circle(face_frame, (x - x1, y - y1), 1, (255, 255, 255), -1)
                except:
                    pass

    # Display current status and speed on the main frame
    cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, f"Speed: {simulated_speed} km/h", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Driver Monitoring", frame)

    # Show face frame in a separate window
    if face_frame is not None and face_frame.size != 0:
        face_frame_resized = cv2.resize(face_frame, (frame.shape[1], frame.shape[0]))
        cv2.imshow("Detected Face", face_frame_resized)

    # Exit on pressing ESC key
    if cv2.waitKey(1) == 27:
        break

# Cleanup
# Releases resources and closes windows.
cap.release()
cv2.destroyAllWindows()

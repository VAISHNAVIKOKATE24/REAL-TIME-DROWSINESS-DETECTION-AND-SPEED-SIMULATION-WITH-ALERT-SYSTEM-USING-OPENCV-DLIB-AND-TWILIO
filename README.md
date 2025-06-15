# Real-Time Drowsiness Detection and Speed Simulation with Alert System Using OpenCV, Dlib, and Twilio


# 🚦 Overview

This project presents a real-time computer vision system to detect driver drowsiness based on Eye Aspect Ratio (EAR), triggering safety mechanisms like:
- Buzzer alert
- Simulated speed reduction
- SMS alerts with IP-based location via Twilio
- Event logging and visualization using Power BI

# 📌 Key Features

- 👁️ Real-time eye monitoring using OpenCV and dlib facial landmark detection
- 💤 Drowsiness classification: **Active**, **Drowsy**, **Sleeping**
- 🔔 Buzzer sound alert on drowsiness/sleep detection
- 📍 SMS alerts sent using Twilio with IP-based geolocation
- 🚗 Simulated speed reduction and dynamic status update
- 📊 Event logging to CSV and Power BI dashboard insights

# 🧠 Technologies Used

- Python 3.10
- OpenCV, dlib, pygame
- Twilio SMS API
- `geocoder` for IP-based location
- Power BI for dashboard
- CSV logging for event tracking

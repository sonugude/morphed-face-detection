# Morphed Face Detection System

## Overview

Morphed Face Detection System is a web-based security application designed to detect and prevent face morphing attacks in facial recognition systems. The project combines facial recognition, face morph detection, and user-friendly web interfaces to improve identity verification and authentication security.

## Technologies Used

* Python
* FastAPI
* OpenCV
* face_recognition
* HTML, CSS, JavaScript
* NumPy
* Deep Learning (Face Morph Detection)
* Git & GitHub

## Features

* Face data collection through webcam
* Face encoding generation and storage
* Real-time face recognition
* Face morph attack detection
* User registration and enrollment
* Image upload and verification
* Web-based interface
* Secure identity verification workflow

## Project Architecture

1. User Registration
2. Face Data Collection
3. Face Encoding Generation
4. Real-Time Face Recognition
5. Face Morph Detection
6. Result Display

## How It Works

The system captures facial images using a webcam or uploaded image. Face encodings are generated and compared with stored reference encodings. Before authentication, the morph detection module analyzes the image to identify potential morphing attacks. The final result indicates whether the face is genuine, recognized, or potentially morphed.

## Challenges Faced

* Managing large face datasets
* Handling unsupported image formats
* Optimizing face recognition performance
* Integrating morph detection with recognition workflow
* Connecting FastAPI backend with frontend interfaces

## What I Learned

* Computer Vision fundamentals
* Facial Recognition systems
* Face Morph Attack Detection
* FastAPI development
* Frontend and Backend integration
* Git and Version Control
* Security-focused application development

## Future Improvements

* Database integration for scalable storage
* Multi-factor authentication
* Cloud deployment
* Enhanced morph detection models
* User activity logging and analytics
* Mobile-friendly interface

## Installation

### Clone the Repository

```bash
git clone https://github.com/sonugude/morphed-face-detection.git
cd morphed-face-detection
```

### Create Virtual Environment

```bash
python -m venv myvenv
```

### Activate Virtual Environment

```bash
myvenv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
uvicorn backend.main:app --reload
```

### Open in Browser

```text
http://127.0.0.1:8000
```

## Project Status

Currently under active development with ongoing improvements in face morph detection accuracy and system performance.

## Author

Sonu Gude

Cybersecurity & Software Development Enthusiast

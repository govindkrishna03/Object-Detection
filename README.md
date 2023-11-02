# YOLO Detection Web App - Detecting people in a video / cctv

This repository contains a web application for object detection using the YOLO (You Only Look Once) deep learning model. The primary purpose of this application is to detect and count people in video streams. The YOLO model has proven to be an efficient and accurate solution for object detection tasks, making it suitable for this project.

## Features

- Detect and count people in video streams.
- User-friendly web interface for easy interaction.

## Prerequisites

Before using the application, ensure you have the following dependencies installed:

- Python 3.8+
- Flask - a Python web framework.
- OpenCV - for image and video stream processing.

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/govindkrishna03/Object-Detection.git
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Download the YOLO pre-trained weights (e.g., YOLOv3) and configuration files and place them in the `yolo` directory.

4. Run the web application:

```bash
python app.py
```

The app should now be accessible in your web browser at `http://localhost:5000`.

## Usage

1. Open the web application in your browser.

2. Choose a video file to process for real-time detection.

3. Click the "Upload" button to display the video with the object detection process.

4. The application will display the processed image or video with bounding boxes around detected people, along with the count.

## Customization

- Modify the HTML and CSS files in the `templates` and `static` directories to change the appearance and layout of the web application.

- Add additional features or object classes to the application as needed.


## Demo
[![Video](https://www.example.com/video-thumbnail.png)](https://drive.google.com/file/d/1pX_I_VvvsuDDVB6BaaTpLLfxRt003ekg/view?usp=sharing)

# YOLOv4 Image Detection API

## Description
This project implements a FastAPI-based API for object detection using the YOLOv4 model. It loads YOLOv4 weights, configuration, and class names, processes input images, applies non-max suppression to filter detections, draws bounding boxes on the image, and returns detection data along with the annotated image encoded in base64.

## Features
- **Object Detection**: Uses the YOLOv4 model to detect objects in images.
- **FastAPI Integration**: Provides a REST API endpoint for image detection.
- **Annotation**: Annotates detected objects with bounding boxes and labels.
- **Image Encoding**: Returns the annotated image as a base64 encoded string.
- **CORS Support**: Configured to allow requests from a React front-end (http://localhost:4000).

## Technologies Used
- Python
- FastAPI
- Uvicorn
- OpenCV
- NumPy
- Base64 encoding

## Installation and Setup

### Requirements
- Python 3.8 or later.
- YOLOv4 weight file (`yolov4.weights`), configuration file (`yolov4.cfg`), and class names file (`coco.names`).

### Installing Dependencies
Install the required libraries using pip:
```sh
pip install fastapi uvicorn opencv-python numpy base64
```

## How to Run
1. Ensure that the `yolov4.weights`, `yolov4.cfg`, and `coco.names` files are available in your working directory.
2. Start the API using Uvicorn:
   ```sh
   uvicorn your_script_name:app --host 127.0.0.1 --port 3000
   ```
   Replace `your_script_name` with the name of your Python file containing the API code.
3. The API will be available at `http://127.0.0.1:3000`.

## API Endpoint

### POST `/detect/`
Receives an image via a POST request and returns detection information along with an annotated image.

#### Request:
- **image**: An image file uploaded as form-data.

#### Response:
- **detections**: A list of detected objects, each containing:
  - **class**: The detected class name.
  - **confidence**: The confidence score for the detection.
  - **box**: Bounding box coordinates in the format `[x, y, w, h]`.
- **annotated_image**: The annotated image with drawn bounding boxes, encoded as a base64 string.

## Project Structure
```
/
|-- your_script_name.py   # Main API implementation file.
|-- yolov4.weights        # YOLOv4 weights file.
|-- yolov4.cfg            # YOLOv4 configuration file.
|-- coco.names            # File containing class names.
|-- README_YOLOv4.md      # Detailed documentation.
```

## GitHub Short Description
"FastAPI-based API for YOLOv4 object detection in images with bounding box annotations and base64 image output."

## License
This project is distributed under the MIT license.

---

Built with FastAPI and YOLOv4 🚀

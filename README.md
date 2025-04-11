# HiDolphin Coffee Detection Project

A real-time coffee detection system using YOLO (You Only Look Once) object detection models, built with Streamlit for an intuitive web interface.

## 🚀 Features

- **Multiple Model Support**: Choose between YOLOv8m and YOLO11n models for detection
- **Flexible Input Sources**: 
  - Image upload and processing
  - Video upload and processing
  - Default sample images and videos
- **Customizable Detection**: Adjustable confidence threshold for detection accuracy
- **Real-time Processing**: Process both images and videos with detection results
- **Results Storage**: Automatically saves processed images and videos with timestamps

## 📋 Prerequisites

- Python 3.x
- Required Python packages:
  - Streamlit
  - Ultralytics (YOLO)
  - OpenCV (cv2)
  - PIL (Python Imaging Library)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd HiDolphin_Detection_Project
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

1. Run the Streamlit app:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Configure the detection:
   - Select your preferred YOLO model (YOLOv8m or YOLO11n)
   - Adjust the confidence threshold using the slider
   - Choose between image or video input
   - Upload your media or use the default samples

4. Click the "检测图片" (Detect Image) or "检测视频" (Detect Video) button to process your input

## 📁 Project Structure

```
HiDolphin_Detection_Project/
├── images/           # Directory for input images
├── videos/           # Directory for input videos
├── results/          # Directory for processed outputs
├── weights/          # Directory for model weights
├── main.py           # Main application file
└── requirements.txt  # Project dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLO team for the object detection models
- Streamlit team for the web framework
- OpenCV for computer vision capabilities
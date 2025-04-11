#import the necessary libraries
import cv2
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO
from PIL import Image
from datetime import datetime

#get the absolute path of the current file
FILE = Path(__file__).resolve()

#get the parent directory of the current file
ROOT = FILE.parent

#add the root path to the sys.path
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

#get the relative path of the ROOT directory with respect to the current file
ROOT = ROOT.relative_to(Path.cwd())

#sources
IMAGE = '图片'
VIDEO = '视频'

SOURCE_LIST = [IMAGE, VIDEO]

#image config
IMAGE_DIR = ROOT / 'images'
IMAGE_DIR.mkdir(exist_ok=True)
DEFAULT_IMAGE = IMAGE_DIR / 'image1.jpg'
DEFAULT_Detected_IMAGE = IMAGE_DIR / 'detectedimage1.jpg'

#video config
VIDEO_DIR = ROOT / 'videos'
VIDEO_DIR.mkdir(exist_ok=True)
VIDEO_DICT = {
    'video 1': VIDEO_DIR / 'RCK137.mp4'
}

#results config
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

#model config
MODEL_DIR = ROOT / 'weights'
MODEL_DIR.mkdir(exist_ok=True)
DETECTION_MODEL_11n = MODEL_DIR / 'yolo11n_coffee_spill_detection.pt'
DETECTION_MODEL_v8m = MODEL_DIR / 'yolo11n_coffee_spill_detection.pt'

#page layout
st.set_page_config(page_title="YOLO11",
                   page_icon=":guardsman:",
                   layout="wide")

st.header("HiDolphi 机器人现磨咖啡检测")

st.sidebar.header("模型配置")

#choose the model
model_type = st.sidebar.radio("选择想要检测的模型",
                             ("YOLOv8m", "YOLO11n"))

#select confidence value
confidence_value = float(st.sidebar.slider("选择置信度阈值",
                                    25,40,100))/100

#select model
if model_type == 'YOLOv8m':
    model_path = Path(DETECTION_MODEL_v8m)
elif model_type == 'YOLO11n':
    model_path = Path(DETECTION_MODEL_11n)

#load the model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"模型加载失败，请检查模型路径是否正确: {model_path}")
    st.error(e)

#image / video config
st.sidebar.header("图片/视频 配置")
source_radio = st.sidebar.radio("选择输入源",
                              SOURCE_LIST)
source_image = None
if source_radio == IMAGE:
    source_image = st.sidebar.file_uploader("上传图片...",
                                           type=['jpg', 'jpeg', 'png'])
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_image is None:
                default_image_path = str(DEFAULT_IMAGE)
                default_image = Image.open(default_image_path)
                st.image(default_image_path, caption='Default Image', use_container_width=True)
            else:
                uploaded_image = Image.open(source_image)
                st.image(source_image, caption='Uploaded Image', use_container_width=True)
        except Exception as e:
            st.error("图片加载失败")
            st.error(e)
    with col2:
        try:
            if source_image is None:
                default_detected_image_path = str(DEFAULT_Detected_IMAGE)
                default_detected_image = Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image', use_container_width=True)
            else:
                if st.sidebar.button("检测图片"):
                    result = model.predict(uploaded_image, conf=confidence_value)
                    boxes = result[0].boxes
                    result_plotted = result[0].plot()[:, :, ::-1]
                    st.image(result_plotted, caption='Detected Image', use_container_width=True)
                    
                    # Save the detected image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_image_path = RESULTS_DIR / f"detected_{timestamp}_{source_image.name}"
                    cv2.imwrite(str(output_image_path), result_plotted)
                    st.success(f"处理后的图片已保存到: {output_image_path}")
                    
                    try:
                        with st.expander("检测结果"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as e:
                        st.error("检测结果显示失败...")
                        st.error(e)
        except Exception as e:
            st.error("图片加载失败...")
            st.error(e)

elif source_radio == VIDEO:
    # Video source selection
    video_source_type = st.sidebar.radio("选择视频来源",
                                       ["默认视频", "上传视频"])
    
    video_path = None  # Initialize video_path
    
    if video_source_type == "默认视频":
        source_video = st.sidebar.selectbox("选择视频...",
                                          VIDEO_DICT.keys())
        if source_video:  # Check if a video is selected
            video_path = VIDEO_DICT.get(source_video)
    else:
        uploaded_video = st.sidebar.file_uploader("上传视频...",
                                                type=['mp4', 'mov'])
        if uploaded_video:
            # Save uploaded video temporarily
            temp_video_path = VIDEO_DIR / f"temp_{uploaded_video.name}"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            video_path = temp_video_path
    
    if video_path:  # Now video_path is properly initialized
        # Display video
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            if video_bytes:
                st.video(video_bytes)
            
            if st.sidebar.button("检测视频"):
                try:
                    video_cap = cv2.VideoCapture(str(video_path))
                    st_frame = st.empty()
                    
                    # Get video properties
                    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
                    
                    # Create output video writer with H.264 codec
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_video_path = RESULTS_DIR / f"detected_{timestamp}_{Path(video_path).name}"
                    # Use 'avc1' codec for better compatibility
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (640, 640))
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    current_frame = 0
                    
                    while video_cap.isOpened():
                        success, image = video_cap.read()
                        if success:
                            current_frame += 1
                            progress_bar.progress(current_frame / total_frames)
                            
                            # Resize image for better performance
                            image = cv2.resize(image, (640, 640))
                            # Run detection
                            result = model.predict(image, conf=confidence_value)
                            # Plot results
                            result_plotted = result[0].plot()
                            # Convert BGR to RGB for saving
                            result_plotted = cv2.cvtColor(result_plotted, cv2.COLOR_BGR2RGB)
                            # Save frame to output video
                            out.write(result_plotted)
                            # Display frame with detection
                            st_frame.image(result_plotted, 
                                         caption='检测结果', 
                                         use_container_width=True, 
                                         channels='RGB')
                        else:
                            video_cap.release()
                            out.release()
                            progress_bar.empty()
                            break
                    
                    # Show success message with saved video path
                    st.success(f"处理后的视频已保存到: {output_video_path}")
                    
                    # Clean up temporary file if it was an upload
                    if video_source_type == "上传视频" and uploaded_video:
                        temp_video_path.unlink()
                        
                except Exception as e:
                    st.error("视频处理失败...")
                    st.error(e)











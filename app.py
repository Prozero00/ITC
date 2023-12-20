import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import time
from collections import Counter
import json
import pandas as pd
from model_utils import get_yolo, color_picker_fn, get_system_stat

st.sidebar.title('Settings')
st.title('Sistem Klasifikasi Sampah')
sample_img = cv2.imread('OIP.jpeg')
FRAME_WINDOW = st.image(sample_img, channels='BGR', width = 640)
cap = None
p_time = 0


path_model_file = 'best.pt'
model = torch.hub.load('.', 'custom', path=path_model_file, source='local',  force_reload=True)

class_labels = model.names

        # device options
if torch.cuda.is_available():
    device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
else:
    device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

options = st.sidebar.radio(
        'Options:', ( 'Image', 'Video', 'Webcam'), index=1)

# Web-cam
if options == 'Webcam':
    cam_options = st.sidebar.selectbox('Webcam Channel',
                                        ('Select Channel', '0', '1', '2', '3'))
    
    if not cam_options == 'Select Channel':
        pred = st.checkbox('Predict Using YOLOv5')
        cap = cv2.VideoCapture(int(cam_options))

confidence = st.sidebar.slider(
        'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

draw_thick = st.sidebar.slider(
    'Draw Thickness:', min_value=1,
    max_value=20, value=3
    )

color_pick_list = []
for i in range(len(class_labels)):
    classname = class_labels[i]
    color = color_picker_fn(classname, i)
    color_pick_list.append(color)

if options == 'Image':
    upload_img_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

    if upload_img_file is not None:
        pred = st.checkbox('Predict Using YOLOv5')
        file_bytes = np.asarray(
            bytearray(upload_img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        FRAME_WINDOW.image(img, channels='BGR', width=640)

        if pred:
            img, current_no_class = get_yolo(img, model, confidence, color_pick_list, draw_thick)
            FRAME_WINDOW.image(img, channels='BGR', width=640)

            # Current number of classes
            class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
            class_fq = json.dumps(class_fq, indent = 4)
            class_fq = json.loads(class_fq)
            df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                
            # Updating Inference results
            with st.container():
                st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
                st.dataframe(df_fq, use_container_width=True)
    
        # Video
if options == 'Video':
    upload_video_file = st.sidebar.file_uploader(
            'Upload Video', type=['mp4', 'avi', 'mkv'])
    if upload_video_file is not None:
        pred = st.checkbox('Predict Using YOLOv5')

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(upload_video_file.read())
        cap = cv2.VideoCapture(tfile.name)

if (cap != None) and pred:
    stframe1 = st.empty()
    stframe2 = st.empty()
    stframe3 = st.empty()
    while True:
        success, img = cap.read()
        if not success:
            st.error(
                f"{options} NOT working\nCheck {options} properly!!",
                icon="🚨"
            )
            break

        img, current_no_class = get_yolo(img, model, confidence, color_pick_list, draw_thick)
        FRAME_WINDOW.image(img, channels='BGR')

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        
        # Current number of classes
        class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
        class_fq = json.dumps(class_fq, indent = 4)
        class_fq = json.loads(class_fq)
        df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
        
        # Updating Inference results
        get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)
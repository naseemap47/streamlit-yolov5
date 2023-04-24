from PIL import ImageColor
import subprocess
import streamlit as st
import psutil
import random
import cv2


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def get_gpu_memory():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory[0]

def color_picker_fn(classname, key):
    color_picke = st.sidebar.color_picker(f'{classname}:', '#ff0003', key=key)
    color_rgb_list = list(ImageColor.getcolor(str(color_picke), "RGB"))
    color = [color_rgb_list[2], color_rgb_list[1], color_rgb_list[0]]
    return color


def get_yolo(img, model, confidence, color_pick_list, draw_thick):
    current_no_class = []
    results = model(img)
    box = results.pandas().xyxy[0]

    for i in box.index:
        xmin, ymin, xmax, ymax, conf, id, class_name = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
            int(box['ymax'][i]), box['confidence'][i], box['class'][i], box['name'][i]
        if conf > confidence:
            plot_one_box([xmin, ymin, xmax, ymax], img, label=class_name,
                            color=color_pick_list[id], line_thickness=draw_thick)
        current_no_class.append([class_name])
    return img, current_no_class


def get_system_stat(stframe1, stframe2, stframe3, fps, df_fq):
    # Updating Inference results
    with stframe1.container():
        st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
        if round(fps, 4)>1:
            st.markdown(f"<h4 style='color:green;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h4 style='color:red;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
    
    with stframe2.container():
        st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
        st.dataframe(df_fq, use_container_width=True)

    with stframe3.container():
        st.markdown("<h2>System Statistics</h2>", unsafe_allow_html=True)
        js1, js2, js3 = st.columns(3)                       

        # Updating System stats
        with js1:
            st.markdown("<h4>Memory usage</h4>", unsafe_allow_html=True)
            mem_use = psutil.virtual_memory()[2]
            if mem_use > 50:
                js1_text = st.markdown(f"<h5 style='color:red;'>{mem_use}%</h5>", unsafe_allow_html=True)
            else:
                js1_text = st.markdown(f"<h5 style='color:green;'>{mem_use}%</h5>", unsafe_allow_html=True)

        with js2:
            st.markdown("<h4>CPU Usage</h4>", unsafe_allow_html=True)
            cpu_use = psutil.cpu_percent()
            if mem_use > 50:
                js2_text = st.markdown(f"<h5 style='color:red;'>{cpu_use}%</h5>", unsafe_allow_html=True)
            else:
                js2_text = st.markdown(f"<h5 style='color:green;'>{cpu_use}%</h5>", unsafe_allow_html=True)

        with js3:
            st.markdown("<h4>GPU Memory Usage</h4>", unsafe_allow_html=True)  
            try:
                js3_text = st.markdown(f'<h5>{get_gpu_memory()} MB</h5>', unsafe_allow_html=True)
            except:
                js3_text = st.markdown('<h5>NA</h5>', unsafe_allow_html=True)

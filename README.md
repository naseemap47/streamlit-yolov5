# streamlit-yolov5
Display predicted Video, Images and webcam using YOLO5 models with Streamlit

## Docker
dockerhub: https://hub.docker.com/repository/docker/naseemap47/streamlit-yolo5

#### 1. Pull Docker Image
```
docker pull naseemap47/streamlit-yolo5
```
#### 2. Change permistion
```
sudo xhost +si:localuser:root
```
#### 3. RUN Docker Image
```
docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --ipc=host --device=/dev/video0:/dev/video0 -p 8502 -it --rm naseemap47/streamlit-yolo5
```

## Streamlit Options
### Modes
 - RTSP
 - Webcam
 - Video
 - Image
 
 ## Sample Streamlit Dashboard Output
 
 [out.webm](https://user-images.githubusercontent.com/88816150/193816239-b351c3d6-1d9a-4820-87b5-0cfec1ad5d90.webm)

 ## StepUp
```
git clone https://github.com/naseemap47/streamlit-yolov5.git
cd streamlit-yolov5
```
Install dependency
```
pip3 install -r requirements.txt
```
Run **Streamlit**
```
streamlit run app.py
```

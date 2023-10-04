# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:48:23 2023

@author: bbman
"""

import cv2
import dlib
import mediapipe as mp

module_path = r"C:\Users\bbman\anaconda3\Lib\site-packages"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(module_path + "/shape_predictor_68_face_landmarks.dat")
face_rec = dlib.face_recognition_model_v1(module_path + "/dlib_face_recognition_resnet_model_v1.dat")

def return_128d_features(img_bgr):

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb, 1)

    # 确保检测到是人脸图像去算特征
    if len(faces) != 0:

        # 存取人臉特徵描述符
        shape = predictor(img_rgb, faces[0])

        # 將人臉特徵描述符轉換成特徵向量
        face_descriptor = face_rec.compute_face_descriptor(img_rgb, shape)

    else:
        face_descriptor = 0

    return face_descriptor

#from sklearn.metrics import silhouette_score
from skimage import transform as trans
#from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
from skimage import io 
#import pandas as pd
import numpy as np
import os, csv, dlib, shutil


# MediaPipe Face Mesh 變數宣告
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#-----使用 RetinaFace 來進行人臉偵測，再利用 scikit-image 來裁切-----

src = np.array([
   [30.2946, 51.6963],
   [65.5318, 51.5014],
   [48.0252, 71.7366],
   [33.5493, 92.3655],
   [62.7299, 92.2041]], dtype=np.float32)


path_name = os.path.abspath(os.getcwd()) + '/'  #記得去掉副檔名
video_path = path_name+'video.mp4'
path = path_name + 'face_inf'

# 獲取影片的 Frames Per Second
video = cv2.VideoCapture(video_path)
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver) < 3 :
  fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
else :
  fps = video.get(cv2.CAP_PROP_FPS)
video.release()
fps = int(round(fps))

print("fps =",fps)

cnt = 0
data = []
imgs = []

cap = cv2.VideoCapture(video_path)

with mp_face_mesh.FaceMesh(max_num_faces = 3, refine_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:

    while cap.isOpened():
      success, img  = cap.read()
      
      if not success:
        print("finish")
        break
    
      else:
          
        h, w, d = img.shape
          
        cnt += 1
        seconds = cnt//fps
        
        # 每秒偵測一次
        if cnt % fps == 0:
          if seconds % 60 == 0: print(seconds/60)
    
          # 使用Dlib函式庫將臉部資訊存入 detections
          imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          results = face_mesh.process(imgRGB)
          
          if results.multi_face_landmarks:
 
              for face_data in results.multi_face_landmarks:
                
                # print('nose_coordinate =', face_info['nose'][0], face_info['nose'][0])
                
                # 儲存切割特徵點
                right_eye = [int((face_data.landmark[386].x*w+face_data.landmark[374].x*w)//2), int((face_data.landmark[386].y*h+face_data.landmark[374].y*h)//2)]
                left_eye = [int((face_data.landmark[159].x*w+face_data.landmark[145].x*w)//2), int((face_data.landmark[159].y*h+face_data.landmark[145].y*h)//2)]
                nose = [int(face_data.landmark[1].x*w), int(face_data.landmark[1].y*h)]
                right_lip = [int(face_data.landmark[291].x*w), int(face_data.landmark[291].y*h)]
                left_lip = [int(face_data.landmark[61].x*w), int(face_data.landmark[61].y*h)]

                face_landmarks = [left_eye, right_eye, nose, left_lip, right_lip]
                
                # 將特徵座標進行對齊
                dst = np.array(face_landmarks, dtype=np.float32).reshape(5, 2)
                tform = trans.SimilarityTransform()
                tform.estimate(dst, src)
                
                # 將圖片進行縮放裁切
                M = tform.params[0:2, :]
                aligned = cv2.warpAffine(imgRGB, M, (112, 112), borderValue=0)
                
                # 將結果進行存取
                im_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                imgs.append(im_rgb)

cap.release()

# 創建資料夾
if os.path.exists(path): shutil.rmtree(path)
os.mkdir(path)

length = len(imgs)
temp_path = path + '/aligned_imgs'

if os.path.exists(temp_path): shutil.rmtree(temp_path)
os.mkdir(temp_path)

# 將影像存入檔案、影像資訊存成CSV檔
for i in range(length):
    name = 'img' + str(i+1) + '.jpg'
    cv2.imencode('.jpg', imgs[i])[1].tofile(path + '/aligned_imgs/' + name)
    #cv2.imwrite(path + '/aligned_imgs/' + name, imgs[i])

# -----Dlib 函式庫取特徵向量-----

path_images_from_our = path + "/aligned_imgs"
 
data, success, not_success = [], [], []
na = 0

# 返回单张图像的 128D 特征
def img_return_128d_features(path_img):
    img_bgr = io.imread(path_img)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb, 1)

    # 确保检测到是人脸图像去算特征
    if len(faces) != 0:

        # 存取人臉特徵描述符
        shape = predictor(img_rgb, faces[0])

        # 將人臉特徵描述符轉換成特徵向量
        face_descriptor = face_rec.compute_face_descriptor(img_rgb, shape)
        success.append(na)

    else:
        face_descriptor = 0
        not_success.append(na)

    return face_descriptor
 
# 读取每个人人脸图像的数据
people = os.listdir(path_images_from_our)
os.listdir()

# 紀錄特徵向量資料
with open(path + "/feature_vector.csv", "w", encoding = 'utf-8-sig', newline="") as csvfile:
    writer = csv.writer(csvfile)

    # 遍歷目標資料夾中的每張影像
    for person in people:
      na = person
      features_128d = img_return_128d_features(path_images_from_our + "/" + person)
      a = features_128d
      if a == 0: continue

      writer.writerow(features_128d)
      
      # 將特徵向量存入'data'
      data.append(features_128d)

print('success :',len(success))
print('not_success :',len(not_success))
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:48:11 2023

@author: bbman
"""

import cv2
import dlib

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
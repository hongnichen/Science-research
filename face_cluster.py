# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:21:55 2023

@author: bbman
"""

def ini():
    import cv2
    import dlib
    import csv
    import os
    import shutil
    import time
    
    import numpy as np
    import mediapipe as mp
    import matplotlib.pyplot as plt
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score   
    
    from face_functions import return_128d_features
    
    from skimage import transform as trans
    
    code_start_time = time.time()
    print("face_cluster program start")
    
    # MediaPipe Face Mesh 參數設定
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    
    path_name = os.path.abspath(os.getcwd()) + '/'  #記得去掉副檔名
    #print(path_name)
    video_path = path_name+'video.mp4'
    path = path_name + 'face_inf'
    cap = cv2.VideoCapture(video_path)
    
    # 宣告變數
    cnt, death_number = 0, 0
    imgs, feature_vector, to_speaker_recognition = [], [], []
    
    # 獲取 FPS
    video = cv2.VideoCapture(video_path)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    fps = int(round(fps))
    
    # 臉部特徵座標
    src = np.array([
       [30.2946, 51.6963],
       [65.5318, 51.5014],
       [48.0252, 71.7366],
       [33.5493, 92.3655],
       [62.7299, 92.2041]], dtype=np.float32)
    
    # 宣告 Dlib 檢測器、預測器和識別模型
    module_path = r"C:\Users\bbman\anaconda3\Lib\site-packages"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(module_path + "/shape_predictor_68_face_landmarks.dat")
    face_rec = dlib.face_recognition_model_v1(module_path + "/dlib_face_recognition_resnet_model_v1.dat")
    
    # 存取特徵向量
    with open(path + "/feature_vector.csv",encoding = 'utf-8-sig', newline='') as csvfile:
    
      rows = csv.reader(csvfile)
      for row in rows:
        feature_vector.append(row)  
    
    # 用輪廓係數法尋找最佳 K 值
    feature_vector = np.array(feature_vector).astype(np.float32)
    sil_score = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(feature_vector)
        sil_score.append(silhouette_score(feature_vector, kmeans.labels_))
    best_k = np.argmax(sil_score) + 2
    #print(best_k)
    
    #best_k = 3
    print('face_cluster_best K =', best_k)
    
    # K-Means分群
    Kmeans = KMeans(n_clusters = best_k, random_state=0, n_init=10)
    Kmeans.fit(feature_vector.astype('double'))
    
    # 使用 MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
    
        # MediaPipe Face Mesh 參數設定 
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        # 逐幀讀取影像
        print("video start")
        while cap.isOpened():
    
            success, img = cap.read()
    
            if not success:
                print("video end")
                break
    
            cnt += 1
            seconds = cnt // fps
            
            h, w, d = img.shape
    
            # 存取臉部特徵
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
    
            if cnt%fps == 0 and seconds%10 == 0:
                if seconds%60 == 0: print(seconds//60, "minutes")
                else: print(seconds%60, "s", end = ' ')
    
            # 有人臉
            if results.multi_face_landmarks:
    
                # if death_number == 5: break
    
                n = 0
    
                if cnt%fps == 0 or cnt%(fps//2) == 0:
    
                    for face_data in results.multi_face_landmarks:
    
                        dx = int(face_data.landmark[308].x * w) - int(face_data.landmark[78].x * w)
                        dy = int(face_data.landmark[15].y * h) - int(face_data.landmark[12].y * h)
                        
                        if dy != 0 and dx / dy <= 4: n += 1
    
                    if n != 1: continue
    
                    for face_data in results.multi_face_landmarks:
    
                        dx = int(face_data.landmark[308].x * w) - int(face_data.landmark[78].x * w)
                        dy = int(face_data.landmark[15].y * h) - int(face_data.landmark[12].y * h)
    
                        if dy == 0 or dx / dy > 4 : continue
                        
                        right_eye = [int((face_data.landmark[386].x*w+face_data.landmark[374].x*w)//2), int((face_data.landmark[386].y*h+face_data.landmark[374].y*h)//2)]
                        left_eye = [int((face_data.landmark[159].x*w+face_data.landmark[145].x*w)//2), int((face_data.landmark[159].y*h+face_data.landmark[145].y*h)//2)]
                        nose = [int(face_data.landmark[1].x*w), int(face_data.landmark[1].y*h)]
                        right_lip = [int(face_data.landmark[291].x*w), int(face_data.landmark[291].y*h)]
                        left_lip = [int(face_data.landmark[61].x*w), int(face_data.landmark[61].y*h)]
    
                        x, y = int(face_data.landmark[10].x*w), int(face_data.landmark[10].y*h)
    
                        face_landmarks = [left_eye, right_eye, nose, left_lip, right_lip]        
    
                        # 將特徵座標進行對齊
                        dst = np.array(face_landmarks, dtype=np.float32).reshape(5, 2)
                        tform = trans.SimilarityTransform()
                        tform.estimate(dst, src)
    
                        # 將圖片進行縮放裁切
                        M = tform.params[0:2, :]    
                        aligned_img = cv2.warpAffine(img_rgb, M, (112, 112), borderValue=0)
    
                        # 將特徵進行存取
                        aligned = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
                        dlib_vector_feature = return_128d_features(aligned)
                        if dlib_vector_feature == 0: continue
    
                        feature = list(dlib_vector_feature)
                        feature_2D = [feature, feature]
    
                        feature_2D = np.array(feature_2D)
    
                        the_group = str(Kmeans.predict(feature_2D.astype('double'))[0])
    
                        cv2.circle(img, (x, y), 6, (255, 0, 255), -1)
                        cv2.putText(img, the_group, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA)
    
                        temp = []
                        temp.append(str(cnt/fps))
                        temp.append(the_group)
    
                        to_speaker_recognition.append(temp)
    
                        # print(cnt/fps, ": group", the_group)
    
            # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            # cv2.imshow("video", img)
            # if cv2.waitKey(1) == ord('q'): break
                       
    cap.release()
    
    # 將資料寫入 to_speaker_recognition.csv
    #if not os.path.exists(path): os.mkdir(path)
    
    
    with open(path + "/face_cluster_data.csv", 'w', encoding='utf-8-sig') as csvfile:
        csvfile.write('秒數,分群\n')
        for info in to_speaker_recognition:
            csvfile.write(info[0] +','+ info[1] + '\n')
    
    code_end_time = time.time()
    
    dtime = int(code_end_time - code_start_time)
    cost_time = str(dtime//60) + " : " + str(dtime%60)
    if dtime%60 == 0:  cost_time += '0'
    print("The program has been running for " + cost_time)
    print("program end")

if __name__ == '__main__':
    ini()
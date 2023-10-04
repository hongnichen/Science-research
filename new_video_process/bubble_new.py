# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 13:45:16 2023

@author: bbman
"""

import cv2, os, csv, dlib, shutil, time

import mediapipe as mp
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from return_feature import return_128d_features
import diarization_match

from PIL import Image, ImageFont, ImageDraw
from skimage import transform as trans

# 快速前置區
n_frames = 15
path_name = os.path.abspath(os.getcwd()) + '/'  #記得去掉副檔名
video = path_name+'video.mp4'
video_path = video

feature_vector = []
csvfile_path = path_name + "face_inf/feature_vector.csv"

# 存取特徵向量
with open(csvfile_path,encoding = 'utf-8-sig', newline='') as csvfile:

    rows = csv.reader(csvfile)
    for row in rows:
        feature_vector.append(row)

# 用輪廓係數法尋找最佳 K 值
feature_vector = np.array(feature_vector).astype(np.float32)
sil_score = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init=10).fit(feature_vector)
    sil_score.append(silhouette_score(feature_vector, kmeans.labels_))
K = np.argmax(sil_score) + 2

#K = 3

# K-Means分群
Kmeans = KMeans(n_clusters = K, random_state=0, n_init=10)
Kmeans.fit(feature_vector.astype('double'))

# MediaPipe Face Mesh 變數宣告
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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

# 創建圖片檔案
img_file = path_name + 'img/'
if not os.path.exists(img_file):os.mkdir(img_file)
else:
  shutil.rmtree(img_file)
  os.mkdir(img_file)
 
# 判斷是不是中文字
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
#print(is_Chinese("中文"))

# 獲取 fps
def get_fps():

    video = cv2.VideoCapture(video_path)
    ret, img = video.read()
    h, w, d = img.shape
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3 : fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else : fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps, w

from moviepy.editor import VideoFileClip

clip = VideoFileClip(video_path)
#print( clip.duration ) # seconds
duration = clip.duration

csv_audio_path = path_name + "classification_results.csv"
# 為影片嵌入字幕
def insert_text():
    
    # 引入字幕資訊內容
    subtitle_inf = pd.read_csv(path_name + "subtitle.csv",encoding='utf-8-sig')
    total_lines = sum(1 for line in open(path_name + "subtitle.csv",encoding='utf-8-sig')) - 1
    
    # 存取臉部和語音分群編號對照
    distinguish_results = [[False]*K for i in range(total_lines)] #K is the cluster number of face cluster
    distinguish_results = diarization_match.Speaker_cluster(K)
    '''
    distinguish_results =  [ [False, True] ,
                            [False, True] ,
                            [False, True] ,
                            [False, True] ,
                            [False, True] ,
                            [False, True] ,
                            [True, False] ,
                            [True, False] ,
                            [True, False] ,
                            [True, False] ,
                            [False, True] , 
                            [False, True] , 
                            [False, True] , 
                            [False, True] , 
                            [False, True] , 
                            [False, True] ]
    '''

    print("bubble_subtitle program start")
    program_start_time = time.time()

    cap = cv2.VideoCapture(video_path)

    
    
    #讀取環境辨識資訊
    audio_inf = pd.read_csv(csv_audio_path, encoding='utf-8-sig')
    audio_total_lines = sum(1 for line in open(csv_audio_path, encoding='utf-8-sig')) - 1
    audio_cnt = 0

    
    
    # 宣告變數
    fps, w = get_fps()
    total_frame_count = int(duration*fps)
    cnt_frames, cnt_distinguish, cnt_OwnerlessText = 0, 0, 0
    id, x, y = 0, 0, 0

    data, old_id, BoxColor = [], [], []
    OPEN_CLOSE = [[False]*n_frames for i in range(K)]
    
    # 設定文字資料
    fontpath = "font/simsun.ttc"
    FontSize = 32
    if FontSize > w/25: FontSize //= 2
    font = ImageFont.truetype(fontpath, FontSize)
    

    # 使用 MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(max_num_faces = 4, refine_landmarks = True, 
                               min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:

        if not cap.isOpened():
            print("Fail to read video")
            exit()

        while True:

            ret, img = cap.read()

            if not ret:
                print("\nThe video is over")
                break

            # 宣告變數
            flag_NewText, flag_NewFace = False, False
            ColorType, TextType, owner = (255, 255, 255), 0, -1
            Get_GroupX = [0]*K
            NewData = []
            

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(imgRGB)

            h, w, d = img.shape

            cnt_frames += 1
            now = cnt_frames / fps
            #seconds = int(now)
            
            
            '''
            # 輸出時間
            if cnt_frames % fps == 0 and seconds % 10 == 0:
                if seconds % 60 == 0: print(seconds // 60, "minutes")
                else: print(seconds % 60, "s")
            '''
            percentage = int(cnt_frames*30 / (total_frame_count+1))
            print("\rProcess: [{}{}] {:06d} / {:06d}".format("#"*percentage, "."*(30-1-percentage), cnt_frames, total_frame_count), end ='')
            
            # 填上情境字幕
            if now > audio_inf.iloc[audio_cnt]['結束時間']/1000:
                #print("audio cnt", audio_cnt)
                audio_cnt += 1
                
            if audio_cnt < audio_total_lines:
                if now >= audio_inf.iloc[audio_cnt]['起始時間']/1000 and now <= audio_inf.iloc[audio_cnt]['結束時間']/1000:
                    # 將文字嵌入影像
                    #print("in time")
                    
                    x = w*1//10
                    y = h*9//10
                    max_x = x
                    audio_1, audio_2, audio_3 = 0, 0, 0
                    if audio_inf.iloc[audio_cnt]['分數一'] > 0.3:
                        audio_1 = 1
                        text_len = 0
                        for i in audio_inf.iloc[audio_cnt]['結果一']:
                            if is_Chinese(i):
                                text_len += FontSize
                            else:
                                text_len += FontSize//2
                        max_x = max(max_x, x+text_len)
                        
                        if audio_inf.iloc[audio_cnt]['結果一'] == 'Speech':
                            audio_1 = 0
                        # 將文字框嵌入
                        if audio_1 == 1:
                            bbox = [x, y, max_x+FontSize, y+FontSize]  #x+FontSize due to '[]', y+FontSize due to first line
                            #print("bbox",bbox)
                            zeros = np.zeros((img.shape), dtype=np.uint8)
                            zeros_mask1 = cv2.rectangle(zeros, (bbox[0], bbox[1]), (bbox[2], bbox[3]),  color=(	200, 200, 0), thickness=-1) #thickness=-1 表示矩形框内颜色填充
                            zeros_mask = np.array(zeros_mask1)
                            img = cv2.addWeighted(img, 1,  zeros_mask, 0.6, 1)
                        if audio_inf.iloc[audio_cnt]['分數二'] > 0.3:
                            max_x = x
                            audio_2 = 1
                            text_len = 0
                            for i in audio_inf.iloc[audio_cnt]['結果二']:
                                if is_Chinese(i):
                                    text_len += FontSize
                                else:
                                    text_len += FontSize//2
                            max_x = max(max_x, x+text_len)
                            bbox = [x, y-FontSize-FontSize//6, max_x+FontSize, y-FontSize//6]  #x+FontSize due to '[]', y+FontSize due to first line
                            #print("bbox",bbox)
                            zeros = np.zeros((img.shape), dtype=np.uint8)
                            zeros_mask1 = cv2.rectangle(zeros, (bbox[0], bbox[1]), (bbox[2], bbox[3]),  color=(	200, 200, 0), thickness=-1) #thickness=-1 表示矩形框内颜色填充
                            zeros_mask = np.array(zeros_mask1)
                            img = cv2.addWeighted(img, 1,  zeros_mask, 0.6, 1)
                            if audio_inf.iloc[audio_cnt]['分數三'] > 0.3:
                                max_x = x
                                audio_3 = 1
                                text_len = 0
                                for i in audio_inf.iloc[audio_cnt]['結果三']:
                                    if is_Chinese(i):
                                        text_len += FontSize
                                    else:
                                        text_len += FontSize//2
                                max_x = max(max_x, x+text_len)
                                bbox = [x, y-FontSize*2-FontSize//3, max_x+FontSize, y-FontSize-FontSize//3]  #x+FontSize due to '[]', y+FontSize due to first line
                                #print("bbox",bbox)
                                zeros = np.zeros((img.shape), dtype=np.uint8)
                                zeros_mask1 = cv2.rectangle(zeros, (bbox[0], bbox[1]), (bbox[2], bbox[3]),  color=(	200, 200, 0), thickness=-1) #thickness=-1 表示矩形框内颜色填充
                                zeros_mask = np.array(zeros_mask1)
                                img = cv2.addWeighted(img, 1,  zeros_mask, 0.6, 1)
                        #print("text")
                        
                       
                        img_pil = Image.fromarray(img)
                        draw = ImageDraw.Draw(img_pil)
                        if audio_1 == 1:
                            draw.text((x, y),'['+ audio_inf.iloc[audio_cnt]['結果一']+']' , font = font, fill = (0, 0, 0))
                        if audio_2 == 1:
                            draw.text((x, y-FontSize-FontSize//6),'['+ audio_inf.iloc[audio_cnt]['結果二']+']' , font = font, fill = (0, 0, 0))
                        if audio_3 == 1:
                            draw.text((x, y-FontSize*2-FontSize//3),'['+ audio_inf.iloc[audio_cnt]['結果三'] +']', font = font, fill = (0, 0, 0))
                        img = np.array(img_pil)

            # 超出當前字幕
            if id < total_lines and now > subtitle_inf.iloc[id]["結束時間"]: 
                id += 1

            # 字幕全部嵌入完畢
            if id >= total_lines:
                cv2.imencode('.jpg', img)[1].tofile(img_file+str(cnt_frames-1)+'.jpg')
                # 輸出影像
                img = cv2.resize(img, (0, 0), fx = 0.75, fy = 0.75)
                cv2.imshow('Video', img)
                if cv2.waitKey(1) == ord('q'): break
                continue

            #==================================================================================================================#
            
            #print("\nOPEN_CLOSE")
            #for arr in OPEN_CLOSE:
            #    print("len =", len(arr))
            #print('\n')
            for i in range(K): del OPEN_CLOSE[i][0]
            
            flag_NewText = now >= subtitle_inf.iloc[id]['起始時間'] and now <= subtitle_inf.iloc[id]['結束時間']
            
            coordinate = []
            SpeakerOpen = []
            
            # 有人臉
            if results.multi_face_landmarks:
                
                cnt_distinguish += 1

                flag_CorrectFace = False
                cnt_speaker = 0
                SpeakerData = []

                # 當前有字幕

                # 遍歷每張臉並記錄資料
                for face_data in results.multi_face_landmarks:

                    # 標示有張嘴者
                    GreenX = int(face_data.landmark[152].x*w)
                    GreenY = int(face_data.landmark[152].y*h)

                    cv2.circle(img, (GreenX, GreenY), 6, (0, 0, 255), -1)

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
                    aligned_img = cv2.warpAffine(imgRGB, M, (112, 112), borderValue=0)

                    # 將特徵進行存取
                    aligned = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
                    dlib_vector_feature = return_128d_features(aligned)
                    if dlib_vector_feature == 0:
                        continue

                    feature = list(dlib_vector_feature)
                    feature_2D = [feature, feature]
                    feature_2D = np.array(feature_2D)

                    the_group = Kmeans.predict(feature_2D.astype('double'))[0]

                    #dx = int(face_data.landmark[308].x * w) - int(face_data.landmark[78].x * w)
                    #dy = int(face_data.landmark[15].y * h) - int(face_data.landmark[12].y * h)
                   
                    #if dy == 0: flag_NOW_OPEN_CLOSE = False
                    #else: flag_NOW_OPEN_CLOSE = dx / dy <= 4
                    flag_NOW_OPEN_CLOSE = True

                    OPEN_CLOSE[the_group].append(flag_NOW_OPEN_CLOSE)

                    flag_OPEN, flag_CLOSE, flag_BEFORE_OPEN_CLOSE = False, False, False

                    for val in OPEN_CLOSE[the_group]:
                        if val == True: flag_OPEN = True
                        else: flag_CLOSE = True
                        if flag_OPEN and flag_CLOSE:
                            flag_BEFORE_OPEN_CLOSE = True
                            break

                    to_SpeakerData = []
                    to_SpeakerData.append(the_group)
                    to_SpeakerData.append([int(face_data.landmark[454].x*w), int(face_data.landmark[152].y*h)])

                    if flag_BEFORE_OPEN_CLOSE or flag_NOW_OPEN_CLOSE: to_SpeakerData.append(True)

                    else: to_SpeakerData.append(False)

                    cv2.circle(img, (int(face_data.landmark[152].x*w), int(face_data.landmark[152].y*h)), 6, (0, 255, 0), -1)
                    
                    SpeakerData.append(to_SpeakerData)

                cnt_speaker = 0
                SpeakerOpen = []

                for val in SpeakerData:

                    Get_GroupX[val[0]] = val[1][0]

                    if val[2] == True: 
                        cnt_speaker += 1
                        SpeakerOpen.append(val)

                #if cnt_speaker >= 2: key = distinguish_results[id]
                Key = distinguish_results[id]

                # 標示分群編號
                for val in SpeakerData: cv2.putText(img, str(val[0]), (val[1][0], val[1][1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                

                '''
                # 當前有字幕且有正確人臉
                if cnt_speaker == 1:
                    
                    owner = SpeakerOpen[0][0]
                    x = SpeakerOpen[0][1][0]
                    y = SpeakerOpen[0][1][1]
                    TextType = 1
                    ColorType = (255, 255, 255)
                    flag_NewFace =True

                    #cv2.putText(img, "Speaking", (10, h-10), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 1, cv2.LINE_AA)
                '''

                    
                
                '''
                # 沒人說話
                if cnt_speaker == 0:

                    x, y = w*4//5, h*9//10
                    TextType = 0
                    ColorType = (255, 255, 255)
                    coordinate.append([-1, x, y, TextType, ColorType, id, 0])
                    #print(f'coordinate append with 0, {cnt_frames}')

                    #cv2.putText(img, "No Speaking", (10, h-10), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 1, cv2.LINE_AA)
                '''

                #elif  cnt_speaker >= 2:
                
                for val in SpeakerOpen:

                    #if val[0] == key:
                    if Key[val[0]]:
                        
                        owner = val[0]
                        x = val[1][0]
                        y = val[1][1]
                        flag_CorrectFace = True
                        TextType = 1
                        ColorType = (255, 255, 255)
                        coordinate.append([owner, x, y, TextType, ColorType, id, 1])
                        #print(f'coordinate append with 1, {cnt_frames}')

                        flag_NewFace  =True

                        #避免卡到環境字幕
                        #cv2.putText(img, "Speaking", (10, h-10), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 1, cv2.LINE_AA)

                        #break
                   
            for i in range(K):
                bol = True
                for val in SpeakerOpen:
                    if val[0]==i: bol = False
                if not bol: continue
                if Key[i]:
                    owner = i
                    flag_CorrectFace = True
                    x, y = w*4//5, h*9//10
                    TextType = 0
                    ColorType = (255, 255, 255)
                    coordinate.append([owner, x, y, TextType, ColorType, id, 2])

            # 沒正確人臉
            if not flag_CorrectFace:

                x, y = w*4//5, h*9//10
                TextType = 0
                ColorType = (255, 255, 255)
                coordinate.append([-1, x, y, TextType, ColorType, id, 3])
                #print(f'coordinate append with 3, {cnt_frames}')
                #cv2.putText(img, "No Correct Face", (10, h-10), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 1, cv2.LINE_AA)


            '''
            # 沒人臉
            else:

                x, y = w*4//5, h*9//10
                TextType = 0
                ColorType = (255, 255, 255)
                coordinate.append([-1, x, y, TextType, ColorType, id, 3])
                #print(f'coordinate append with 3, {cnt_frames}')
            '''

            
            for i in range(K):
                tLEN = len(OPEN_CLOSE[i])
                if tLEN == n_frames: continue
                if tLEN < n_frames:
                    tD = n_frames - tLEN
                    for j in range(tD):
                        OPEN_CLOSE[i].append(False)
                elif tLEN > n_frames:
                    tD = tLEN - n_frames
                    for j in range(tD):
                        del OPEN_CLOSE[i][0]

            # 標示當前座標位置
            for i in range(len(coordinate)):
                x = coordinate[i][1]
                y = coordinate[i][2]
                cv2.circle(img, (x, y), 6, (255, 0, 0), -1)
            
            if flag_NewText:
                for i in old_id:
                    if id == i:
                        flag_NewText = False
                        break                        

            # 有新字幕
            if flag_NewText:

                old_id.append(id)

                for i in range(len(coordinate)):
                    #print(coordinate)
                    owner = coordinate[i][0]
                    x = coordinate[i][1]
                    y = coordinate[i][2]
                    TextType = coordinate[i][3]
                    ColorType = coordinate[i][4]
                    
                    # 算每行字數和總行數 and 將字幕分段
                    line_len = int(w-x)
                    Text = subtitle_inf.iloc[id]['字幕']
    
                    # 
                    # 
                    textlist = list(Text)
                    Text_len = 0
                    text_len = 0
                    next_line = 0
                    i=0
                    while i < len(textlist):
                        if is_Chinese(textlist[i]):
                            text_len += FontSize
                            Text_len += FontSize
                        else:
                            text_len += FontSize//2
                            Text_len += FontSize//2
                        if text_len > w-x:
                            if i+1<len(textlist) and textlist[i+1].isalpha() and textlist[i-1].isalpha() and is_Chinese(textlist[i+1]) == False and  is_Chinese(textlist[i-1]) == False: 
                                textlist.insert(i-1, '-')
                            textlist.insert(i, '\n')
                            #print('換行')
                            next_line += 1
                            text_len = 0
                            if is_Chinese(textlist[i]): text_len += FontSize
                            else: text_len += FontSize//2
                                
  
                        i += 1
                    nline = next_line+1
    
                    textmark = ''
                    for i in range(0,len(textlist)): textmark+=textlist[i]
                    #print(f"text = {textmark}")
                    if h-y < nline*FontSize: y = h-nline*FontSize
    
                    #目前狀況：無法讓最後一行字幕框契合最後一個字的位置 因為是一個長方形不然很難填充 解決辦法可能是一行字就畫一個長方形 好麻煩
                    #還有如果換行前最後一個字是英文字下一個是中文 那字幕框會填到畫面邊邊而不是英文字的一半那邊
    
                    # 算 xr 和 yr
                    if Text_len>=line_len: xr = x+line_len
                    else: xr = x+Text_len
                    yr = y + nline*FontSize
    
                    NewData.append([True, owner, id, y])
    
                    #將資料存入 data
                    to_data = []
                    
                    to_data.append([id, 0]) #0
                    to_data.append([x, y]) #1
                    to_data.append(textmark) #2
                    to_data.append([x, y, xr, yr]) #3
                    to_data.append(TextType) #4
                    to_data.append(owner) #5
                
                    data.append(to_data)
    
                    if TextType == 0: cnt_OwnerlessText += 1
                    BoxColor.append(ColorType)

            '''
            if flag_NewFace:

                # 算每行字數和總行數 and 將字幕分段
                line_len = int(w-x)
                Text = subtitle_inf.iloc[id]['字幕']

                # 
                textlist = list(Text)
                Text_len = 0
                text_len = 0
                next_line = 0
                for i in range(len(Text)):
                    if is_Chinese(textlist[i+next_line]):
                        text_len += FontSize
                        Text_len += FontSize
                    else:
                        text_len += FontSize//2
                        Text_len += FontSize//2
                    if text_len > w-x:
                        textlist.insert(next_line+i, '\n')
                        next_line += 1
                        text_len = 0
                nline = next_line+1

                textmark = ''
                for i in range(0,len(textlist)): textmark+=textlist[i]
                if h-y < nline*FontSize: y = h-nline*FontSize

                #目前狀況：無法讓最後一行字幕框契合最後一個字的位置 因為是一個長方形不然很難填充 解決辦法可能是一行字就畫一個長方形 好麻煩
                #還有如果換行前最後一個字是英文字下一個是中文 那字幕框會填到畫面邊邊而不是英文字的一半那邊

                # 算 xr 和 yr
                if Text_len>=line_len: xr = x+line_len
                else: xr = x+Text_len
                yr = y + nline*FontSize
                '''

            #==================================================================================================================#

            # -----0-----------------1-----------------2-----3---------------------------------4----------5----
            # data([id, frames_cnt], [text_x, text_y], text, [box_x1, box_y1, box_x2, box_y2], text_type, owner)

            #NewData(flag, owner, id, y)

            # 逐個把文字框嵌入影像並上移
            i = 0
            while i < len(data):

                # 將字幕添加至新臉
                if flag_NewFace and data[i][4] == 0:
                #if flag_NewFace:
                    for val in SpeakerData:
                        if distinguish_results[data[i][0][0]][val[0]] and data[i][5]==val[0]:
                            owner = val[0]
                            box_dx = data[i][3][2] - data[i][3][0]
                            box_dy = data[i][3][3] - data[i][3][1]
                            MoveUp = data[i][0][1] * 2
        
                            data[i][0][1] = 0
                            data[i][1][0], data[i][1][1] = x, y - MoveUp
                            data[i][3][0], data[i][3][1], data[i][3][2], data[i][3][3] = x, y - MoveUp, x + box_dx, y + box_dy - MoveUp
                            data[i][4] = 1
                            data[i][5] = owner
        
                            cnt_OwnerlessText -= 1
        
                            NewData.append([True, owner, data[i][0][0], data[i][1][1]])

                for newdata in NewData:
                    if newdata[0]:
    
                        ty = newdata[3]
                        tGroup = newdata[1]
                        tID = newdata[2]
                        bottom = -1
    
                        for val in data:
                            #print(val)
                            if val[5] == tGroup and val[3][3] >= bottom:
                                if val[0][0] == tID: 
                                    continue
                                bottom = val[3][3]
    
    
                        if bottom >= ty:
                            all_MoveUp = bottom - ty + 10
                            for i in range(len(data)):
                                if data[i][5] == tGroup and data[i][0][0] != tID:
                                    data[i][1][1] -= all_MoveUp
                                    data[i][3][1] -= all_MoveUp
                                    data[i][3][3] -= all_MoveUp
                                    
                NewData = []

                # 將文字框嵌入
                bbox = [data[i][3][0], data[i][3][1], data[i][3][2], data[i][3][3]]
                zeros = np.zeros((img.shape), dtype=np.uint8)
                zeros_mask1 = cv2.rectangle(zeros, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255, 255, 255), thickness=-1)
                zeros_mask = np.array(zeros_mask1)
                img = cv2.addWeighted(img, 1,  zeros_mask, 0.6, 1)
                
                # 將文字框、文字座標上移 
                data[i][1][1] -= 2
                data[i][3][1] -= 2
                data[i][3][3] -= 2
                data[i][0][1] += 1
                if Get_GroupX[data[i][5]] != 0: 
                    box_dx = Get_GroupX[data[i][5]] - data[i][3][0]
                    data[i][1][0] = Get_GroupX[data[i][5]]
                    data[i][3][0] += box_dx
                    data[i][3][2] += box_dx

                flag_del = False

                # 碰到頂或達到數量刪除
                if data[i][1][1] <= 0: 
                    del data[i]
                    flag_del = True
                elif data[i][0][1] >= 150: 
                    del data[i]
                    flag_del = True

                if not flag_del: i += 1

            # 將文字嵌入影像
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            for val in data: draw.text((val[1][0], val[1][1]),  val[2], font = font, fill = (0, 0, 0))
            img = np.array(img_pil)

            cv2.imencode('.jpg', img)[1].tofile(img_file+str(cnt_frames-1)+'.jpg')

            # 輸出影像
            img = cv2.resize(img, (0, 0), fx = 0.75, fy = 0.75)
            cv2.imshow('Video', img)
            if cv2.waitKey(1) == ord('q'): break

            if id >= total_lines: break

    cap.release()
    cv2.destroyAllWindows()

    program_end_time = time.time()

    dtime = int(program_end_time - program_start_time)
    
    cost_time = str(dtime//60) + " : " + str(dtime%60)
    if dtime%60 == 0:  cost_time += '0'
    print("cnt_frames =", cnt_frames)
    print("The program has been running for " + cost_time)
    print("program end")

    return fps

if __name__ == '__main__':
    insert_text()
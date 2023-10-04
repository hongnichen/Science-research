# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 10:39:50 2023

@author: bbman
"""
from __future__ import division
import os

#快速前置區
Speaker_cluster_Type = 3  # 0->kmeans, 1->diarization, 2->color_mark, 3->color_diarizaiton
separate_audio = False
#nSpeakers = 2

# 引入 time 模組
import time
# 開始測量
start = time.time()

# 結束測量
#end = time.time()

# 輸出結果
#print("執行時間：%f 秒" % (end - start))

path_name = os.path.abspath(os.getcwd()) + '/'  #記得去掉副檔名
video = path_name+'video.mp4'
audiopath = path_name+'audio.wav'
audiotype = 'wav' #如果wav、mp4其他格式参看pydub.AudioSegment的API
#cut_len = 300
#cut_dB = -35

#import globals
#globals.initialize()
#globals.nSpeaker = nSpeakers

from moviepy.editor import AudioFileClip

try:
    from pydub import AudioSegment
    print('讀入音頻')
    sound_test = AudioSegment.from_file(audiopath, format=audiotype)
except:
    # 開始測量
    audio_start = time.time()

    #將影片音訊另外生成出來
    
    print("讀取失敗 額外生成影片音訊")
    from moviepy.editor import VideoFileClip
    Video = VideoFileClip(path_name+'video.mp4')
    audio = Video.audio
    audio.write_audiofile(path_name+'audio.wav')
    Video.reader.close()
    
    # 結束測量
    audio_end = time.time()

    # 輸出結果
    print("生成音訊時間：%f 秒" % (audio_end - audio_start))

#'''
#主程式差不多開始

try:
    print('讀入字幕')
    import pandas as pd
    test_inf = pd.read_csv(path_name + 'subtitle.csv', encoding='utf-8-sig')
except:
    print("讀取失敗 生成影片字幕")
    print('開始辨識字幕')
    #import wav2csv

try:
    print("讀取背景音效")
    import pandas as pd
    test_inf = pd.read_csv(path_name + 'classification_results.csv', encoding='utf-8-sig')
except:
    print("沒有音效辨識結果檔案，開始辨識環境音效")
    import audio_classification

    audio_classification.ini()

if separate_audio:
    print('分離環境噪音')
    import spr
    spr.separate_audio(audiopath, '2stems', path_name+'audio')

#生成人臉與聲音分群資訊
import face_cluster
face_cluster.ini()

fps = 0
#import face_recognition
#fps = face_recognition.insert_text(TEXT, TIME)

if Speaker_cluster_Type == 0:
    import bubble_subtitles
    fps = bubble_subtitles.insert_text()   #該函式回傳fps
elif Speaker_cluster_Type == 1:
    import bubble_new
    fps = bubble_new.insert_text()  #該函式回傳fps
elif Speaker_cluster_Type == 2:
    import add_subtitles_v2
    fps = add_subtitles_v2.insert_text()  #該函式回傳fps
elif Speaker_cluster_Type == 3:
    import color_subtitles_diarization
    fps = color_subtitles_diarization.insert_text()  #該函式回傳fps

if fps == 0:
    print("error, video fps = 0")

#'''
#fps = 30    

import cv2
import os 
import numpy as np

# 引入 time 模組
#import time
# 開始測量
video_start = time.time()

print("開始合成影片")
#path = path_name+'img/*.jpg'
path = path_name+'img/'
modify_nosound_path = path_name+'video_modified_nosound.mp4'

result_name = modify_nosound_path
frame_list = os.listdir(path)

frame_list.sort(key=lambda x:int(x[:-4]))

print("frame count: ",len(frame_list))
#print(frame_list)
#fps = 30                                                                              #記得改掉
#shape = cv2.imread(path+frame_list[0]).shape # delete dimension 3   #path不能有中文
shape = cv2.imdecode(np.fromfile(path+frame_list[0], dtype=np.uint8), 1).shape

size = (shape[1], shape[0])
print("frame size: ",size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(result_name, fourcc, fps, size)

for idx in range(len(frame_list)):
    #frame = cv2.imread(path+frame_list[idx]) #不能中文path
    frame = cv2.imdecode(np.fromfile(path+frame_list[idx], dtype=np.uint8), 1)

    current_frame = idx+1
    total_frame_count = len(frame_list)
    percentage = int(current_frame*30 / (total_frame_count+1))
    print("\rProcess: [{}{}] {:06d} / {:06d}".format("#"*percentage, "."*(30-1-percentage), current_frame, total_frame_count), end ='')
    out.write(frame)

out.release()
cv2.destroyAllWindows()
print('')
# 結束測量
video_end = time.time()
# 輸出結果
print("影片合成花費：%f 秒" % (video_end - video_start))
print("Finish making video\n")



modify_path = path_name+'video_modified.mp4'
modify_nosound_path = path_name+'video_modified_nosound.mp4'
# 引入 time 模組
#import time
# 開始測量
video_start = time.time()
from moviepy.editor import VideoFileClip
#from moviepy.editor import AudioFileClip
print("為影片加上聲音")
VIDEO = VideoFileClip(modify_nosound_path)  # 讀取影片
Audio = AudioFileClip(audiopath)        # 讀取音樂
 
new_video = VIDEO.set_audio(Audio)
print("combine over")
new_video.write_videofile(modify_path)
print('save done')

VIDEO.reader.close()
new_video.reader.close()
Audio.close()
# 結束測量
video_end = time.time()
# 輸出結果
print("影片添加音訊花費：%f 秒" % (video_end - video_start))

# 結束測量
end = time.time()

# 輸出結果
print("影片處理花費總時間：%f 秒" % (end - start))

print("情境化字幕影片處理大功告成！！！")





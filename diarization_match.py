# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 23:35:08 2023

@author: bbman
"""

shift_sec = 0  #切割時間平移秒數

import os
path_name = os.path.abspath(os.getcwd()) + '/'  #記得去掉副檔名
audiopath = path_name+'audio.wav'        #人聲/音檔
audiotype = 'wav' #如果wav、mp4其他格式参看pydub.AudioSegment的API
had_face_inf = True
'''
#將影片音訊另外生成出來
from moviepy.editor import VideoFileClip
Video = VideoFileClip(path_name+'video.mp4')
audio = Video.audio
audio.write_audiofile(path_name+'audio.wav')
'''


import numpy as np
#from scipy.io.wavfile import read
#import librosa
#from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
#import shutil
frame_size = 256
frame_shift = 128
#sr = 16000


def Speaker_cluster(face_cluster_k):
    
    print('\n開始語者分段標記')
    #gpustat --w 查看GPU狀況
    # install from develop branch
    #!pip install -qq https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip
    
    # 1. visit hf.co/pyannote/speaker-diarization and hf.co/pyannote/segmentation and accept user conditions (only if requested)
    # 2. visit hf.co/settings/tokens to create an access token (only if you had to go through 1.)
    # 3. instantiate pretrained speaker diarization pipeline
    
    # 引入 time 模組
    import time
    # 開始測量
    start = time.time()
    
    #import os 
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #（代表仅使用第0，1号GPU）
    
    
    import torch
    from pyannote.audio import Pipeline
    
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token="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") 
    
    #pipeline.freeze({"segmentation": {'min_cluster_size': 10}})
    pipeline = pipeline.to(torch.device(0))  #using gup cuda:0  使用gpustat --w 查看
    
    
    # 4. apply pretrained pipeline
    try:
        diarization = pipeline(path_name + 'audio/vocals.wav',
                               num_speakers=face_cluster_k)
    except:
        print('使用無分離噪音音檔')
        diarization = pipeline(audiopath,
                           num_speakers=face_cluster_k)
    
    #num_speakers=3
    #min_speakers=2, max_speakers=5
    
    audio_k = 0
    # 5. print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        if(int(speaker[9]) > audio_k): audio_k = int(speaker[9])
        #print(turn.start, turn.end, _, speaker)
    # start=0.2s stop=1.5s speaker_0
    # start=1.8s stop=3.9s speaker_1
    # start=4.2s stop=5.7s speaker_0
    # ...
    
    #diarization
    
    audio_k += 1
    #print(f"audio k = {audio_k}")
    
    # 結束測量
    end = time.time()
    # 輸出結果
    print("語者分段標記執行時間：%f 秒" % (end - start))
    
    print("開始語者標記")
    # 引入 time 模組
    #import time
    # 開始測量
    #start = time.time()
    
    import pandas as pd 
    subtitle_inf = pd.read_csv(path_name + "subtitle.csv", encoding='utf-8-sig')
    
    #distinguish_results = np.zeros((total_lines), dtype=np.int32)
    diarization_results = [[False]*audio_k for i in range(total_lines)]
    
    for i in range(total_lines):
        start_s = subtitle_inf.iloc[i]['起始時間']
        end_s = subtitle_inf.iloc[i]['結束時間']
        cmp = np.zeros((audio_k), dtype=np.float64)
        #print('now',i)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            #print(turn.start, turn.end, start_s, end_s)
            if turn.end < start_s: continue
            if turn.start > end_s: break
            
            #print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            #print(turn.start, turn.end, _, speaker)
            #if(speaker[9]=='0'): 
            #    print('yes')
            cmp[int(speaker[9])] += (min(end_s, turn.end) - max(start_s, turn.start)) / (end_s - start_s)
            #print((min(end_s, turn.end) - max(start_s, turn.start)) / (end_s - start_s))
            
        for j in range(audio_k):
            #rint(f"{j} in cmp = {cmp[j]}")
            if cmp[j] >= 0.5:
                diarization_results[i][j] = True
                
    
    if not had_face_inf:
        return diarization_results
    
    
    #diarization_results = [[False, True],[True, True],[False, True],[False, True],[False, True],[False, True],[True, False],[True, False],[True, False],[True, False],[False, True],[False, True],[False, True],[False, True],[True, False],[True, False],[True, False]]
    
    # 開始測量
    start = time.time()
    print("match face and chunks")
    #f.write("match face and chunks\n")
    #import pandas as pd 
    face_inf = pd.read_csv(path_name + "face_inf/face_cluster_data.csv",encoding='utf-8-sig')
    face_total_lines = sum(1 for line in open(path_name + "face_inf/face_cluster_data.csv",encoding='utf-8-sig')) - 1
    #results = [[False]*face_cluster_k for i in range(total_lines)]
    results = [-1]*total_lines  #字幕有比較多的人臉第幾群
    #配對秒數
    i = 0
    j = 0
    flag = True
    flag2 = True
    while i < (face_total_lines):
        while face_inf.iloc[i]['秒數'] > subtitle_inf.iloc[j]['結束時間']:
            j += 1
            #print("下一個字幕", j)
            #f.write("下一個字幕 ")
            #f.write(str(j))
            #f.write('\n')
            if j >= total_lines:
                flag = False
                break
        if not flag:
            break
        while face_inf.iloc[i]['秒數'] < subtitle_inf.iloc[j]['起始時間']:
            i += 1
            if i >= face_total_lines:
                flag2 = False
                break
        #print("正確的i", i)
        #f.write("正確的i ")
        #f.write(str(i))
        #f.write('\n')
        if not flag2:
            break
        u = i
        while face_inf.iloc[u]['秒數'] <= subtitle_inf.iloc[j]['結束時間']:
            u += 1
            if u >= face_total_lines:
                break
        u -= 1
        #f.write("此時的u為"+str(u)+' ')
        #if u == i:
        #    results[j] = face_inf.iloc[i]['分群']   
        cluster = [0]*face_cluster_k
        while i <= u:
            cluster[int(face_inf.iloc[i]['分群'])] += 1
            i += 1
        #i-=1
        Max = 0
        label = -1
        for y in range(face_cluster_k):
            if cluster[y] >  Max:
                Max = cluster[y]
                label = y
        results[j] = label
        #print(j, "包含第", label, "人臉比較多此時i到", i)
        #f.write(str(j))
        #f.write("包含第")
        #f.write(str(label))
        #f.write("人臉比較多 此時的i到")
        #f.write(str(i))
        #f.write('\n')
        
        #i += 1
        
    cluster = np.zeros((audio_k, face_cluster_k), dtype = np.int32)  #聲音群有多少人臉群
    Max = [0]*audio_k
    #nsound = [0]*audio_k
    for i in range(total_lines):   #逐群判斷
        #nsound[kmeans.labels_[i]] += 1
        if not results[i] == -1:
            for j in range(audio_k):
                if diarization_results[i][j]:
                    cluster[j][results[i]] += 1
    
        
    #n_face_used = [0]*face_cluster_k
    for i in range(audio_k):
        MMax = 0
        label = -1
        for j in range(face_cluster_k):
            print(f"audio {i} cluster had {cluster[i][j]} in face {j}")
            if cluster[i][j] > MMax:
                MMax = cluster[i][j]
                label = j
        Max[i] = label
        print("聲音第", i, "群有比較多的人臉第", label, "群")
        #f.write("聲音第")
        #f.write(str(i))
        #f.write("群有比較多的人臉第")
        #f.write(str(label))
        #f.write('群\n')
        
    match_results = [[False]*face_cluster_k for i in range(total_lines)]
        
    for i in range(total_lines):
        for j in range(audio_k):
            if diarization_results[i][j]:
                if Max[j] != -1:
                    match_results[i][Max[j]] = True
            #print(i, '因為分群位置被指派為', results[i])
            #f.write(str(i))
           # f.write(" 因為分群位置被指派為")
            #f.write(str(results[i]))
            #f.write('\n')
            
    #f.close()
    # 結束測量
    end = time.time()
    
    # 輸出結果
    print("語者標記花費：%f 秒" % (end - start))
            
    return match_results
    
total_lines = sum(1 for line in open(path_name + "subtitle.csv",encoding='utf-8-sig')) - 1 #此函數會連第一行標題一起算 因此要減一
if __name__ == '__main__':
    had_face_inf = True
    
    face_cluster_k = 3
    distinguish_results = [[False]*face_cluster_k for i in range(total_lines)]
    distinguish_results = Speaker_cluster(face_cluster_k)
    #print(distinguish_results)
    #print('best K :', best_k)
    #print("The lables for", len(distinguish_results), "speech segmentation belongs to the clusters below:")
    if had_face_inf:
        print('臉部配對結果')
    else:
        print("聲音分群結果")
    for i in range(len(distinguish_results)):
        print(i, distinguish_results[i], "")
        #print(distinguish_results[i], end = ",")
        
        
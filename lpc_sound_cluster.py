# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:37:21 2023

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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import shutil
frame_size = 256
frame_shift = 128
#sr = 16000


def Speaker_cluster(face_cluster_k):
    from lpc import get_lpcfeatures
    #f = open('results.txt', 'w', encoding = "utf-8-sig")
    #f.write("聲音分群結果\n")
    #data = []
    # 引入 time 模組
    import time
    # 開始測量
    start = time.time()
    
    from pydub import AudioSegment
    print('讀入音頻')
    sound = AudioSegment.from_file(audiopath, format=audiotype)
    lens = sound.duration_seconds
    #lens = 300  #取前5分鐘測試
    sound = sound[:lens*1000]
    
    import pandas as pd 
    subtitle_inf = pd.read_csv(path_name + "subtitle.csv",encoding='utf-8-sig')
    # 獲取某一行某一列的值
    #print(df.iloc[0]['字幕'])
    #計算字幕數量
    #global total_lines
    #total_lines = sum(1 for line in open(path_name + "subtitle.csv",encoding='utf-8-sig')) - 1 #此函數會連第一行標題一起算 因此要減一
    
    vq_features_lpc = np.zeros((total_lines, 240), dtype=np.float32)
    #vq_features = np.zeros((total_lines, 12), dtype=np.float32)
    save_path = path_name + '/save_chunks'
    print("save_chunks")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    save_path = save_path + '/'
    
    for i in range(total_lines):
        new = sound[subtitle_inf.iloc[i]['起始時間']*1000 + shift_sec*1000:subtitle_inf.iloc[i]['結束時間']*1000 + shift_sec*1000]
        new.export(save_path + str(i) + '.wav', format=audiotype)
        '''
        #mfcc
        (fs,s) = read(save_path + str(i) + '.wav')
        s = s.astype(float)
        s = s.sum(axis=1) / 2 #將左右聲道混為同一聲道，因為librosa暫不支援單聲道以外的分析
        tempAudio = s
        mfccs = librosa.feature.mfcc(tempAudio, fs, n_mfcc=12, hop_length=frame_shift, n_fft=frame_size)
        mfccs = mfccs / mfccs.max()
        vq_code = np.mean(mfccs, axis=1)
        vq_features[i, :] = vq_code.reshape(1, vq_code.shape[0])
        '''
        #lpc
        vq_features_lpc[i, :] = get_lpcfeatures(save_path + str(i) + '.wav', 15, 16)
    #
    where_are_NaNs = np.isnan(vq_features_lpc)
    vq_features_lpc[where_are_NaNs] = 0.
    where_are_infs = np.isinf(vq_features_lpc)
    vq_features_lpc[where_are_infs] = 0.
    #print(vq_features)
    
    print('將聲音分群')
    mx_k = 1 + 10
    if total_lines <= 10: mx_k = total_lines-1
    if mx_k >= 2:
        K = range(2, mx_k)
        square_error = []
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(vq_features_lpc)
            # = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
            square_error.append(kmeans.inertia_)
    elif mx_k==1:
        square_error = []
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(vq_features_lpc)
        square_error.append(kmeans.inertia_)
    
    '''
    #自己選分群數
    kmeans_list = [KMeans(n_clusters=k, random_state=46, n_init=10).fit(data) for k in range(1, mx_k)]
    x = [model.inertia_ for model in kmeans_list]
    y = []
    
    # Elbow Method
    for i in range(1, mx_k): y.append(i)
    plt.plot(y, x, 'o-')
    plt.title('Elbow Method')
    plt.show()
    print('\n')
    '''
    if mx_k > 2:
        from sklearn.metrics import silhouette_score
        # Silhouette Coefficient
        sil_score = []
        for k in range(2, mx_k):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(vq_features_lpc)
            sil_score.append(silhouette_score(vq_features_lpc, kmeans.labels_))
        best_k = np.argmax(sil_score) + 2
        
        plt.title('sound_cluster_Silhouette Coefficient')
        plt.plot(range(2,mx_k), sil_score, 'o-')
        plt.show()
        
        print('\nbest K :', best_k)
    
    
        plt.figure('Kmeans Number of clusters evaluate')
        plt.plot(K, square_error, "bo-")
        plt.title('the number of clusters')
        plt.xlabel("Number of clusters")
        plt.ylabel("SSE For each step")
        plt.ylim(0, square_error[0]*1.5)
        plt.grid(True)
        plt.show()
    
        #k_n = input("Please input the best K value: ")
        kmeans = KMeans(int(best_k), random_state=0).fit(vq_features_lpc)
    elif mx_k==1:
        best_k = 2
        print('\nbest K :', best_k)
    print("分群結束")
    
    '''
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=2, min_samples=2).fit(vq_features_lpc)
    return clustering.labels_
    
    kmeans = clustering
    '''
    
    # 結束測量
    end = time.time()
    # 輸出結果
    print("語者分群花費：%f 秒" % (end - start))
    
    #for i in range(total_lines):
    #    f.write(str(i)+"對應")
    #    f.write(str(kmeans.labels_[i]))
    #    f.write("\n")
    
    if not had_face_inf:
        return kmeans.labels_
    #print("The lables for", len(kmeans.labels_), "speech segmentation belongs to the clusters below:")
    #for i in range(len(kmeans.labels_)):
    #    print(i,kmeans.labels_[i], "")
    
    # 開始測量
    start = time.time()
    print("match face and chunks")
    #f.write("match face and chunks\n")
    face_inf = pd.read_csv(path_name + "face_inf/face_cluster_data.csv",encoding='utf-8-sig')
    face_total_lines = sum(1 for line in open(path_name + "face_inf/face_cluster_data.csv",encoding='utf-8-sig')) - 1
    results = [-1]*total_lines
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
        i-=1
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
        
        i += 1
        
    cluster = np.zeros((best_k, face_cluster_k), dtype = np.int32)  #人臉群有多少聲音群
    Max = [0]*best_k
    #nsound = [0]*best_k
    for i in range(len(kmeans.labels_)):   #逐群判斷
        #nsound[kmeans.labels_[i]] += 1
        if not results[i] == -1:
            cluster[kmeans.labels_[i]][results[i]] += 1
    
        
    #n_face_used = [0]*face_cluster_k
    for i in range(best_k):
        MMax = -1
        label = -1
        for j in range(face_cluster_k):
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
        
    for i in range(total_lines):
        if results[i] == -1:
            results[i] = Max[kmeans.labels_[i]]
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
            
    return results
    
total_lines = sum(1 for line in open(path_name + "subtitle.csv",encoding='utf-8-sig')) - 1 #此函數會連第一行標題一起算 因此要減一
if __name__ == '__main__':
    had_face_inf = False
    
    face_cluster_k = 2
    distinguish_results = np.zeros((total_lines), dtype=np.int32)
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
        
        
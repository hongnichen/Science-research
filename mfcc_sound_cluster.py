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
from scipy.io.wavfile import read
import librosa
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import shutil
frame_size = 256
frame_shift = 128
#sr = 16000


def Speaker_cluster(face_cluster_k):
    #from lpc import get_lpcfeatures
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
    
    #vq_features_lpc = np.zeros((total_lines, 240), dtype=np.float32)
    vq_features = np.zeros((total_lines, 12), dtype=np.float32)
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
        '''
    #
    where_are_NaNs = np.isnan(vq_features)
    vq_features[where_are_NaNs] = 0.
    where_are_infs = np.isinf(vq_features)
    vq_features[where_are_infs] = 0.
    #print(vq_features)
    
    print('將聲音分群')
    mx_k = 1 + 10
    if total_lines <= 10: mx_k = total_lines-1
    if mx_k >= 2:
        K = range(2, mx_k)
        square_error = []
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(vq_features)
            # = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
            square_error.append(kmeans.inertia_)
    elif mx_k==1:
        square_error = []
        kmeans = KMeans(n_clusters=2, random_state=0).fit(vq_features)
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
            kmeans = KMeans(n_clusters=k, random_state=0).fit(vq_features)
            sil_score.append(silhouette_score(vq_features, kmeans.labels_))
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
        kmeans = KMeans(int(best_k), random_state=0).fit(vq_features)
    elif mx_k==1:
        best_k = 2
        print('\nbest K :', best_k)
    print("分群結束")
    
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
            cluster[face_inf.iloc[i]['分群']] += 1
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
        
        
'''
#lpc 新莫彩曦雙人
C:/Users/bbman/Desktop/陳恩泓/資優班/科學研究/音訊研究/new_video_process/sound_cluster.py
best K : 2
分群結束
語者分群花費：6.840070 秒
聲音分群結果
0 0 
1 1 
2 0 
3 0 
4 0 
5 0 
6 0 
7 0 
8 0 
9 0 
10 0 
11 0 
12 0 
13 0 
14 0 
15 0 
16 0 
17 0 
18 0 
19 0 
20 0 
21 0 
22 0 
23 0 
24 0 
25 0 
26 0 
27 0 
28 0 
29 0 
30 0 
31 0 
32 0 
33 0 
34 1 
35 0 
36 0 
37 0 
38 0 
39 0 
40 0 
41 1 
42 1 
43 0 
44 0 
45 0 
46 0 
47 0 
48 1 
49 0 
50 0 
51 1 
52 0 
53 1 
54 0 
55 1 
56 0 
57 1 
58 0 
59 0 
60 0 
61 0 
62 0 
63 0 
64 0 
65 1 

#lpc
best K : 3
分群結束

語者分群花費：61.931489 秒
聲音分群結果
0 0 
1 0 
2 0 
3 0 
4 0 
5 1 
6 0 
7 0 
8 0 
9 1 
10 2 
11 1 
12 0 
13 1 
14 1 
15 1 
16 1 
17 1 
18 0 
19 0 
20 1 
21 1 
22 0 
23 0 
24 1 
25 2 
26 1 
27 1 
28 0 
29 0 
30 0 
31 1 
32 0 
33 0 
34 0 
35 0 
36 1 
37 0 
38 0 
39 1 
40 1 
41 1 
42 1 
43 1 
44 0 
45 0 
46 1 
47 0 
48 1 
49 1 
50 1 
51 0 
52 0 
53 1 
54 0 
55 0 
56 1 
57 1 
58 0 
59 0 
60 2 
61 2 
62 2 
63 0 
64 1 
65 1 
66 2 
67 2 
68 1 
69 2 
70 2 
71 2 
72 2 
73 2 
74 2 
75 2 
76 1 
77 2 
78 1 
79 2 
80 2 
81 1 
82 1 
83 1 
84 1 
85 1 
86 1 
87 1 
88 1 
89 2 
90 1 
91 2 
92 1 
93 0 
94 2 
95 2 
96 0 
97 2 
98 1 
99 1 
100 2 
101 1 
102 0 
103 1 
104 1 
105 1 
106 1 
107 1 
108 1 
109 2 
110 0 
111 2 
112 1 
113 2 
114 0 
115 0 
116 2 
117 0 
118 1 
119 1 
120 1 
121 1 
122 2 
123 1 
124 2 
125 2 
126 2 
127 1 
128 0 
129 0 
130 0 
131 0 
132 0 
133 0 
134 0 
135 1 
136 2 
137 2 
138 1 
139 1 
140 2 
141 1 
142 1 
143 2 
144 1 
145 0 
146 2 
147 0 
148 1 
149 1 
150 1 
151 0 
152 0 
153 0 
154 0 
155 1 
156 2 
157 0 
158 0 
159 1 
160 2 
161 0 
162 2 
163 0 
164 1 
165 1 
166 2 
167 0 
168 2 
169 2 
170 0 
171 0 
172 1 
173 1 
174 2 
175 1 
176 1 
177 0 
178 0 
179 0 
180 2 
181 1 
182 2 
183 1 
184 1 
185 0 
186 0 
187 1 
188 2 
189 0 
190 0 
191 1 
192 2 
193 2 
194 1 
195 1 
196 2 
197 2 
198 2 
199 1 
200 0 
201 0 
202 0 
203 2 
204 2 
205 2 
206 0 
207 0 
208 1 
209 0 
210 0 
211 0 
212 2 
213 0 
214 0 
215 2 
216 1 
217 0 
218 2 
219 1 
220 2 
221 1 
222 1 
223 1 
224 0 
225 0 
226 1 
227 1 
228 2 
229 0 
230 2 
231 1 
232 1 
233 1 
234 2 
235 0 
236 1 
237 1 
238 0 
239 1 
240 2 
241 0 
242 1 
243 2 
244 0 
245 1 
246 0 
247 0 
248 2 
249 0 
250 0 
251 2 
252 1 
253 1 
254 0 
255 0 
256 1 
257 0 
258 1 
259 1 
260 1 
261 1 
262 2 
263 1 
264 0 
265 0 
266 0 
267 0 
268 0 
269 1 
270 2 
271 0 
272 0 
273 0 
274 0 
275 2 
276 1 
277 1 
278 1 
279 2 
280 2 
281 2 
282 2 
283 2 
284 0 
285 0 
286 0 
287 1 
288 0 
289 1 
290 0 
291 1 
292 1 
293 2 
294 1 
295 0 
296 1 
297 2 
298 0 
299 2 
300 0 
301 2 
302 2 
303 0 
304 2 
305 0 
306 2 
307 2 
308 2 
309 2 
310 1 
311 2 
312 2 
313 1 
314 2 
315 0 
316 0 
317 1 
318 1 
319 2 
320 1 
321 2 
322 1 
323 1 
324 1 
325 0 
326 1 
327 1 
328 0 
329 0 
330 2 
331 1 
332 2 
333 1 
334 2 
335 2 
336 1 
337 2 
338 1 
339 2 
340 1 
341 2 
342 2 
343 2 
344 1 
345 2 
346 0 
347 0 
348 0 
349 2 
350 1 
351 2 
352 2 
353 0 
354 0 
355 2 
356 0 
357 2 
358 2 
359 1 
360 0 
361 2 
362 1 
363 2 
364 0 
365 1 
366 1 
367 1 
368 1 

audio - shift
best K : 3
The lables for 208 speech segmentation belongs to the clusters below:
0 0 
1 1 
2 2 
3 0 
4 0 
5 0 
6 1 
7 0 
8 0 
9 1 
10 0 
11 0 
12 0 
13 0 
14 0 
15 0 
16 1 
17 1 
18 0 
19 0 
20 0 
21 0 
22 1 
23 0 
24 1 
25 1 
26 1 
27 0 
28 0 
29 0 
30 0 
31 0 
32 0 
33 0 
34 1 
35 2 
36 0 
37 0 
38 0 
39 0 
40 0 
41 0 
42 0 
43 0 
44 0 
45 0 
46 0 
47 0 
48 0 
49 0 
50 0 
51 0 
52 0 
53 0 
54 0 
55 0 
56 0 
57 0 
58 1 
59 0 
60 0 
61 0 
62 0 
63 1 
64 0 
65 1 
66 1 
67 0 
68 2 
69 0 
70 0 
71 2 
72 2 
73 0 
74 0 
75 0 
76 0 
77 0 
78 0 
79 1 
80 2 
81 0 
82 0 
83 0 
84 0 
85 1 
86 1 
87 0 
88 1 
89 1 
90 0 
91 0 
92 0 
93 0 
94 0 
95 0 
96 1 
97 1 
98 0 
99 0 
100 1 
101 0 
102 2 
103 1 
104 0 
105 0 
106 1 
107 0 
108 0 
109 2 
110 0 
111 0 
112 0 
113 0 
114 0 
115 0 
116 0 
117 0 
118 0 
119 1 
120 2 
121 2 
122 1 
123 1 
124 0 
125 2 
126 2 
127 2 
128 2 
129 2 
130 2 
131 2 
132 0 
133 0 
134 0 
135 0 
136 1 
137 0 
138 0 
139 0 
140 0 
141 0 
142 0 
143 0 
144 0 
145 0 
146 0 
147 0 
148 2 
149 1 
150 0 
151 0 
152 1 
153 1 
154 1 
155 0 
156 0 
157 0 
158 1 
159 1 
160 1 
161 1 
162 1 
163 1 
164 2 
165 0 
166 0 
167 0 
168 0 
169 0 
170 0 
171 0 
172 0 
173 0 
174 1 
175 1 
176 1 
177 0 
178 0 
179 0 
180 2 
181 0 
182 0 
183 2 
184 0 
185 0 
186 0 
187 2 
188 0 
189 2 
190 0 
191 1 
192 2 
193 2 
194 0 
195 1 
196 1 
197 1 
198 0 
199 0 
200 0 
201 0 
202 0 
203 1 
204 1 
205 2 
206 2 
207 2 
'''
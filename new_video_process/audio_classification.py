# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#print("Hello")
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from scipy.io import wavfile

'''
file_name = '/content/drive/MyDrive/資訊/科展/影片情境化字幕實現探討/測試影片/'
video_name = '新博恩單人'
file_name = file_name + video_name
audio_file_name = file_name + '.wav'
try:
  sound_test = wavfile.read(audio_file_name)
except:
  #將影片音訊另外生成出來
  from moviepy.editor import VideoFileClip
  Video = VideoFileClip(file_name + ".mp4")
  Audio = Video.audio
  Audio.write_audiofile(audio_file_name)
'''

def ini():
    print("成功獲得音效標籤")


# 引入 time 模組
import time
# 開始測量
start = time.time()

import os
path_name = os.path.abspath(os.getcwd()) + '/'  #記得去掉副檔名
audio_file_name = path_name + "audio.wav"
csv_audio_path = path_name + "classification_results.csv"
# Customize and associate model for Classifier
base_options = python.BaseOptions(model_asset_path="D:/Download/yamnet_audio_classifier_with_metadata.tflite")
options = audio.AudioClassifierOptions(
    base_options=base_options, max_results=3)


# Create classifier, segment audio clips, and classify
with audio.AudioClassifier.create_from_options(options) as classifier:
  sample_rate, wav_data = wavfile.read(audio_file_name)
  audio_clip = containers.AudioData.create_from_array(
      wav_data.astype(float) / np.iinfo(np.int16).max, sample_rate)
  classification_result_list = classifier.classify(audio_clip)
  
  #print("audio clip:",audio_clip)
  print(len(classification_result_list))
  #assert(len(classification_result_list) == 5)
  with open(csv_audio_path, 'w', encoding='utf-8-sig') as csvfile:
    csvfile.write('起始時間,結束時間,結果一,分數一,結果二,分數二,結果三,分數三\n')
    # Iterate through clips to display classifications
    timestamp = 0
    for idx in range(len(classification_result_list)):
      classification_result = classification_result_list[idx]
      top_category = classification_result.classifications[0].categories[0]
      second_category = classification_result.classifications[0].categories[1]
      third_category = classification_result.classifications[0].categories[2]
      top_category.category_name = top_category.category_name.replace(",", "/")
      second_category.category_name = second_category.category_name.replace(",", "/")
      third_category.category_name = third_category.category_name.replace(",", "/")
      #print(f'Timestamp {timestamp}: {top_category.category_name} ({top_category.score:.2f})')
      #print(f'Timestamp {timestamp}:second results is {second_category.category_name} ({second_category.score:.2f})')
      #print(f'Timestamp {timestamp}:third results is {third_category.category_name} ({third_category.score:.2f})')
      csvfile.write(str(timestamp) + ',' + str(timestamp+975) + ',' + top_category.category_name + ',' + str(top_category.score) + ',' + second_category.category_name + ',' + str(second_category.score) + ',' + third_category.category_name + ',' + str(third_category.score) + '\n')
      timestamp += 975
      
# 結束測量
end = time.time()
print()
# 輸出結果
print("音效辨識花費：%f 秒" % (end - start))

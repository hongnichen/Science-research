# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:28:58 2023

@author: bbman
"""
import librosa
from spleeter.separator import Separator
import soundfile as sf
import os

#前置區
path_name = os.path.abspath(os.getcwd()) + '/'  #記得去掉副檔名
video = path_name+'video.mp4'
audiopath = path_name+'audio.wav'
audiotype = 'wav' #如果wav、mp4其他格式参看pydub.AudioSegment的API

from moviepy.editor import AudioFileClip
try:
    from pydub import AudioSegment
    print('讀入音頻')
    sound_test = AudioSegment.from_file(audiopath, format=audiotype)
except:


    #將影片音訊另外生成出來
    
    print("讀取失敗 額外生成影片音訊")
    from moviepy.editor import VideoFileClip
    Video = VideoFileClip(path_name+'video.mp4')
    audio = Video.audio
    audio.write_audiofile(path_name+'audio.wav')
    Video.reader.close()
    
    
#wav_file = r"C:/Users/bbman/Desktop/陳恩泓/資優班/科學研究/音訊研究/蔡英文訪談/test/s5/crossing1.wav"
#save_path = "environment5"
#separate_mode = '2stems'

#開始分離
def separate_audio(wav_file, separate_mode, save_path):
    print("分離音訊")
    separator = Separator('spleeter:'+separate_mode)
    wav, sr = librosa.load(wav_file,sr=44100)
    wav = wav.reshape(-1,1)
    prediction = separator.separate(wav)
    other = prediction['accompaniment'][:,0]
    vocals = prediction['vocals'][:,0]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    sf.write(save_path+"/vocals.wav" , vocals,44100, 'PCM_24')
    sf.write(save_path+"/accompaniment.wav" , other,44100, 'PCM_24')
    print("save succeed")
    
if __name__ == "__main__":
    separate_audio(audiopath, "2stems",
                   path_name+'audio/')
    
    


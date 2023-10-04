import os
#快速前置區
path_name = os.path.abspath(os.getcwd()) + '/'  #記得去掉副檔名
video = path_name+'video.mp4'
audiopath = path_name+'audio.wav'
audiotype = 'wav' #如果wav、mp4其他格式参看pydub.AudioSegment的API
#@markdown 使用哪一種辨識模型（small:快/普通，medium:慢/精準）
modelType = "small" #@param ["small", "medium", "large"]

"""
#將影片音訊另外生成出來
from moviepy.editor import VideoFileClip
Video = VideoFileClip(path_name+'video.mp4')
audio = Video.audio
audio.write_audiofile(path_name+'audio.wav')
"""

filename = audiopath #@param {type:"string"}
output_path = path_name
outputFilename = 'subtitle'
#@markdown 語音的語言代碼
lang = 'Chinese' #@param ["Chinese", "English", "Japanese", "Korean", "自動判斷"]
#@markdown 輸出為哪一種格式（.srt:字幕檔、.txt:純文字檔）
outputFormat = 'srt' #@param ['srt', 'txt']

#@markdown 是否全部辨識完成，立即下載字幕檔
start_downloading_immediately = False #@param { type: 'boolean' }
#@markdown 是否即時顯示語音辨識結果
verbose = True #@param { type: 'boolean' }
#@markdown ---

import whisper
import torch
from whisper.utils import get_writer
import re
from datetime import timedelta

print("whisper開始辨識語音")
# 引入 time 模組
import time
# 開始測量
start = time.time()

# GPU or CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def transcribe(filename, output_path, outputFilename, outputFormat) :
  model = whisper.load_model(modelType, device=DEVICE)
  if lang=="自動判斷" :
    print('auto detect language')
    result = model.transcribe(filename, fp16=False, verbose=verbose)    
  else :
    result = model.transcribe(filename, fp16=False, verbose=verbose, language=lang)
  saveToFile(result, output_path, outputFilename, outputFormat)

def saveToFile(result, output_path, filename, fileType) :
  file_writer = get_writer(fileType, output_path)
  file_writer(result, filename)


# 定義函式來讀取SRT檔案和解析字幕
def parse_srt_file(file_path):
    # 打開SRT檔案
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        # 讀取SRT檔案內容
        content = f.read()
    
    # 使用正則表達式從SRT檔案中解析出字幕的起始時間、結束時間和文本內容
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)\n\n'
    matches = re.findall(pattern, content, flags=re.DOTALL)
    
    # 將解析出的起始時間、結束時間和文本內容轉換為字典，方便後續處理
    subtitles = []
    for match in matches:
        subtitle = {
            'index': int(match[0]),
            'start_time': match[1],
            'end_time': match[2],
            'text': match[3]
        }
        subtitles.append(subtitle)
    
    return subtitles

# 定義函式來將時間字串轉換為秒數
def time_to_seconds(time_str):
    time_obj = timedelta(hours=int(time_str[0:2]), minutes=int(time_str[3:5]), seconds=int(time_str[6:8]), milliseconds=int(time_str[9:]))
    return time_obj.total_seconds()

#output_path="C:/Users/bbman/Desktop/陳恩泓/資優班/科學研究/音訊研究/movie_process"
#outputFilename='subtitle'

transcribe(filename, output_path, outputFilename, outputFormat)

subtitles = parse_srt_file(output_path + '/' + outputFilename + '.srt')
strings = ""

print("儲存影片字幕檔(csv)")
with open(output_path + '/' + outputFilename + '.csv', 'w', encoding='utf-8-sig') as csvfile:
  csvfile.write('字幕號碼,字幕,起始時間,結束時間\n')
  for subtitle in subtitles:
    number = subtitle['index']
    start_time = time_to_seconds(subtitle['start_time'])
    end_time = time_to_seconds(subtitle['end_time'])
    text = subtitle['text']
    strings += text
    csvfile.write(str(number) + ',' + text + ',' + str(start_time) + ',' + str(end_time) + '\n')
print(strings)
print('字幕辨識完成')
# 結束測量
end = time.time()

# 輸出結果
print("字幕辨識花費%f 秒" % (end - start))
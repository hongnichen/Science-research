# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:32:44 2023

@author: bbman
"""
import numpy as np

def EUDistance(d,c):
    
    # np.shape(d)[0] = np.shape(c)[0]
    n = np.shape(d)[1]
    p = np.shape(c)[1]
    distance = np.empty((n,p))
    
    if n<p:
        for i in range(n):
            copies = np.transpose(np.tile(d[:,i], (p,1)))
            distance[i,:] = np.sum((copies - c)**2,0)
    else:
        for i in range(p):
            copies = np.transpose(np.tile(c[:,i],(n,1)))
            distance[:,i] = np.transpose(np.sum((d - copies)**2,0))
            
    distance = np.sqrt(distance)
    return distance
            

def lbg(features, M):
    eps = 0.01
    codebook = np.mean(features, 1)
    distortion = 1
    nCentroid = 1
    while nCentroid < M:
        
        #double the size of codebook
        new_codebook = np.empty((len(codebook), nCentroid*2))
        if nCentroid == 1:
            new_codebook[:,0] = codebook*(1+eps)
            new_codebook[:,1] = codebook*(1-eps)
        else:    
            for i in range(nCentroid):
                new_codebook[:,2*i] = codebook[:,i] * (1+eps)
                new_codebook[:,2*i+1] = codebook[:,i] * (1-eps)
        
        codebook = new_codebook
        nCentroid = np.shape(codebook)[1]
        D = EUDistance(features, codebook)
        
        
        while np.abs(distortion) > eps:
       	    #nearest neighbour search
            prev_distance = np.mean(D)
            nearest_codebook = np.argmin(D,axis = 1)
            #print 'nearest neighbour',nearest_codebook
        
            #cluster vectors and find new centroid
            #print np.shape(np.mean(features[:, np.where(nearest_codebook == 0)],2))
            for i in range(nCentroid):
                codebook[:,i] = np.mean(features[:,np.where(nearest_codebook == i)], 2).T #add along 3rd dimension
          
            #replace all NaN values with 0  
            codebook = np.nan_to_num(codebook)    
            #print 'this codebook', codebook
            D = EUDistance(features, codebook)
            distortion = (prev_distance - np.mean(D))/prev_distance
            #print 'distortion' , distortion
            
    
    #print 'final codebook', codebook, np.shape(codebook)
    return codebook

def autocorr(x):
    n = len(x)
    variance = np.var(x)
    x = x - np.mean(x)
    r = np.correlate(x, x, mode = 'full')[-n:] #n numbers from last index (l-n to l)
    result = r/(variance*(np.arange(n, 0, -1)))
    return result
    
  
def createSymmetricMatrix(acf,p):
    R = np.empty((p,p))
    for i in range(p):
        for j in range(p):
            R[i,j] = acf[np.abs(i-j)]
    return R
   
def lpc(s,fs,p):
    
    #divide into segments of 25 ms with overlap of 10ms
    nSamples = np.int32(0.025*fs)
    overlap = np.int32(0.01*fs)
    nFrames = np.int32(np.ceil(len(s)/(nSamples-overlap)))
    #print(nFrames)
    
    #zero padding to make signal length long enough to have nFrames
    padding = ((nSamples-overlap)*nFrames) - len(s)
    if padding > 0:
        signal = np.append(s, np.zeros(padding))
    else:
        signal = s
    segment = np.empty((nSamples, nFrames))
    start = 0
    for i in range(nFrames):
        segment[:,i] = signal[start:start+nSamples]
        start = (nSamples-overlap)*i
    
    #calculate LPC with Yule-Walker    
    lpc_coeffs = np.empty((p, nFrames))
    for i in range(nFrames):
        acf = autocorr(segment[:,i])
#        plt.figure(1)
#        plt.plot(acf)
#        plt.xlabel('lags')
#        plt.ylabel('Autocorrelation coefficients')
#        plt.axis([0, np.size(acf), -1, 1])
#        plt.title('Autocorrelation function')
#        break
        r = -acf[1:p+1].T
        R = createSymmetricMatrix(acf,p)
        lpc_coeffs[:,i] = np.dot(np.linalg.inv(R),r)
        lpc_coeffs[:,i] = lpc_coeffs[:,i]/np.max(np.abs(lpc_coeffs[:,i]))
             
    return lpc_coeffs

def get_lpcfeatures(audio_path, orderLPC, nCentroid):
    from scipy.io.wavfile import read
    #sound = AudioSegment.from_file(r"C:\Users\bbman\Desktop\陳恩泓\資優班\科學研究\音訊研究\movie_process\save_chunks\3.wav", format="wav")
    #lens = sound.duration_seconds
    #sound = sound[:lens*1000] #如果文件较大，先取前3分钟测试，根据测试结果，调整参数
    (fs,s) = read(audio_path)
    lpc_coeff = lpc(s, fs, orderLPC)
    codebooks = lbg(lpc_coeff, nCentroid)
    #print(codebooks)
    lpc_features = codebooks.reshape(orderLPC*nCentroid)
    #print(lpc_features)
    return lpc_features

if __name__ == '__main__':
    orderLPC = 15
    nCentroid = 16
    audio_path = r"C:\Users\bbman\Desktop\我\資優班\科學研究\影片情境化字幕實現探討\new_video_process\save_chunks\4.wav"
    features = get_lpcfeatures(audio_path, orderLPC, nCentroid)
    print(features.shape)
    
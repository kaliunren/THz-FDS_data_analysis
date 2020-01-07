# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 16:38:31 2018

@author: kdyre
"""
import os
import matplotlib.pyplot as plt  
import numpy as np  
import scipy
from scipy.signal import hilbert

from scipy.optimize import fsolve




plt.close('all')




'''
def delete_line(filename,newname,del_line):#读取文件删除第del_line行并建立新文件
    with open(filename, 'r') as old_file:
        with open(newname, 'w+') as new_file:
    
            current_line = 0
     
            # 定位到需要删除的行
            while current_line < (del_line - 1):
                old_file.readline()
                current_line += 1
     
            # 当前光标在被删除行的行首，记录该位置
            seek_point = old_file.tell()
     
            # 设置光标位置
            new_file.seek(seek_point, 0)
     
            # 读需要删除的行，光标移到下一行行首
            old_file.readline()
             
            # 被删除行的下一行读给 next_line
            next_line = old_file.readline()
     
            # 连续覆盖剩余行，后面所有行上移一行
            while next_line:
                new_file.write(next_line)
                next_line = old_file.readline()
     
            # 写完最后一行后截断文件，因为删除操作，文件整体少了一行，原文件最后一行需要去掉
            new_file.truncate()
           
            # ...more code
            
#删除第一行的文字
'''
def ana(filename):
   # delete_line(filename,"temp.txt",1)
    filenameout=filename
   # filename="temp.txt"  
    
    data = np.loadtxt(filename,skiprows=1) # 按照float类型读入数据
    data =data[:,[2,1]]
    data =data[np.lexsort(data[:,::-1].T)]
    data_num = np.unique(data[:,0],return_index=True,axis=0)
    data = data[data_num[1],:]
    
    I=data[:,1]#光电流
    Fact=data[:,0]#实际频率    
    
    StandardFreq=np.arange(0,2000,0.014)
    
    inter=scipy.interpolate.UnivariateSpline(Fact,I,s=0,ext=1)(StandardFreq)
    
    #plt.plot(StandardFreq,inter)
    
    Her=hilbert(inter)
    Amp=abs(Her).copy()
    #Ang=np.unwrap(np.angle(Her))
    
    plt.figure(1)
    
    plt.plot(StandardFreq,Amp)
    
    
    #单边谱转双边谱
    Her2=Her.copy()
    Her2[1:-1]=Her2[1:-1]/2
    Cpx_Data=np.append(Her2,np.conj(np.flipud(Her2[1:-1])))
    
    '''
    #双边谱保存，没有存频率
    Complex2_Hilbert=np.column_stack((np.real(Cpx_Data),np.imag(Cpx_Data)))
    np.savetxt("Complex2_Hilbert.txt",Complex2_Hilbert)
    #保存单边谱
    Complex1_Hilbert=np.column_stack((StandardFreq,np.real(Her),np.imag(Her)))
    np.savetxt("Complex1_Hilbert.txt",Complex1_Hilbert)
    '''
    Amp_Hilbert=np.column_stack((StandardFreq/1000,abs(Her)))
    np.savetxt("R-"+filenameout[0:-4]+"-Amp_Hilbert-origin.txt",Amp_Hilbert)
  
    L=len(StandardFreq)
    Time=np.fft.ifft(L*np.conj(Cpx_Data))
    #Time=np.fft.ifft(L*Cpx_Data)
    
    Fs=0.014   #频率间隔
    T=1/Fs
    t=np.arange(0,len(Cpx_Data),1)/len( Cpx_Data)*T
    
    plt.figure(5)
    plt.plot(np.real(Time))
    
    plt.figure(7)
    plt.plot(np.imag(Time))
    #删除虚部，虚部很小
    Time=np.real(Time)
    
    '''
    Time_Hilbert=np.column_stack((t,Time))
    np.savetxt("Time_Hilbert.txt",Time_Hilbert)
    '''
    
    
    #时域置零
    #平滑
    DT=t[2]-t[1]
    Down=2.26
    NDown=int(Down/DT)
    Up=2.37
    NUp=int(Up/DT)
    
    Time[0:NDown]=0
    Time[NUp::]=0
    
    L=len(Time)
    Freq=np.fft.fft(Time)
    FreqP2 = Freq.copy()/L
    FreqP1 = FreqP2[0:int(L/2)+1].copy()#？？？？？？？？？？？？？？？
    FreqP1[1:-1] = 2*FreqP1[1:-1]
    FreqP1= FreqP1*2
    
    plt.figure(6)
    plt.plot(abs(Her))
    plt.plot(abs(FreqP1))
    
    Smooth_Amp_Hilbert=np.column_stack((StandardFreq/1000,abs(FreqP1)))
    np.savetxt("R-"+filenameout[0:-4]+"-Smooth_Amp_Hilbert-origin.txt",Smooth_Amp_Hilbert)
    print("com")
    ########################
    #os.remove("temp.txt")

filesname=[]
for file in os.listdir():  
    #if os.path.splitext(file)[1]==".txt":
    if file=="A042201-T26.5H40.2-P1392.93.txt":
        if file[0]=="A":
            filesname.append(file)
            ana(file)
            
for file in os.listdir():             
    if os.path.exists("temp.txt"):
        os.remove("temp.txt")
        #os.unlink(my_file)

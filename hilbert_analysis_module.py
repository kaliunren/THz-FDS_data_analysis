"""


@author: kdyren 孔德胤
@email: kdyren@hotmail.com

modified at 20191015
original file: hilbert_analysis.py


模块化内容

操作流程：
1. pre_process
2. MHilbert
3. Smooth_TimeDomain
4. OpticalValue (Uncheckecd)


Modeifed at 20190320：Fix bug

"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import hilbert

from scipy.optimize import fsolve

d = 0.000522
c = 299792458

plt.close('all')


def delete_line(filename, newname, del_line):  # 读取文件删除第del_line行并建立新文件
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


def pre_process(filename='', ifInterp=False, Freq=np.arange(0, 2000, 0.014)):
    "频率单位： GHz  返回插值前，插值后数据：return data, data_interpoint"
    data = np.loadtxt(filename, skiprows=1)  # 按照float类型读入数据
    data = data[:, [2, 1]]
    # data = data[np.lexsort(data[:,::-1].T)]
    data_num = np.unique(data[:, 0], return_index=True, axis=0)  # 去除重复频率的点
    data = data[data_num[1], :].copy()

    if ifInterp:
        data_interpoint = scipy.interpolate.UnivariateSpline(data[:, 0],
                                                             data[:, 1],
                                                             s=0,
                                                             ext=1)(Freq)
        data_interpoint = np.column_stack((Freq, data_interpoint))
        return data_interpoint
    else:
        return data


def MHilbert(data):
    "输入的为频率和信号，但没有返回频率。"
    HilbertResult = hilbert(data[:, 1])  # 希尔伯特变换结果

    A = abs(HilbertResult).copy()
    φ = np.unwrap(np.angle(HilbertResult)).copy()

    # 单边谱转双边谱
    HilbertResult2 = HilbertResult.copy()
    HilbertResult2[1:-1] = HilbertResult2[1:-1] / 2
    HilbertResult2 = np.append(HilbertResult2,
                               np.conj(np.flipud(HilbertResult2[1:-1])))

    return HilbertResult, HilbertResult2


def Freq2Time(CpxData, dFreq=0.014, method="IRFFT"):
    "dFreq:频率步长，用于计算时间  irfft: single band \n rfft: double band, 影响CpxData"
    L = len(CpxData)
    T = 1 / dFreq
    if method == "IFFT":

        Time = np.fft.ifft(L * np.conj(CpxData))
        # Time=np.pad(Time,(0,len(CpxData)-len(Time)),'constant')

        t = np.arange(0, len(CpxData), 1) / len(CpxData) * T

        # 删除虚部，虚部很小
        # TimeRef=np.real(TimeRef)
        # TimeSam=np.real(TimeSam)
        return t, Time
    elif method == "IRFFT":
        Time = np.fft.irfft(L * np.conj(CpxData))
        # Time=np.pad(Time,(0,len(CpxData)-len(Time)),'constant')
        t = np.arange(0, len(Time), 1) / len(Time) * T

        # 删除虚部，虚部很小
        # TimeRef=np.real(TimeRef)
        # TimeSam=np.real(TimeSam)
        return t, Time


def Smooth_TimeDomain(HilbertResult, dFreq=0.014, Down=2.28, Up=2.38):
    "默认输入单边谱 对Hilbert后的数据进行平滑，时域版本; 返回频域，不含频率"
    t, Time = Freq2Time(CpxData=HilbertResult, dFreq=dFreq)
    # 时域置零
    DT = t[2] - t[1]

    N_Down = int(Down / DT)
    N_Up = int(Up / DT)

    Time[0:N_Down] = 0
    Time[N_Up::] = 0

    L = len(Time)
    Freq = np.fft.fft(Time)  # 转回频域
    FreqP2 = Freq / L
    FreqP1 = FreqP2[0:int(L / 2) + 1].copy()
    FreqP1[1:-1] = 2 * FreqP1[1:-1]

    return FreqP1, FreqP2


def OpticalValue(freq, FreqSamP1, FreqRefP1, thick=1, DD=17 * np.pi):
    "频率单位：GHz 获取光学常数,\n return T,N"
    PhaseD = np.angle(FreqSamP1) - np.angle(FreqRefP1)  # 反过来减是由于低频相位翻转
    PhaseD = np.unwrap(PhaseD)
    PhaseD = PhaseD + DD
    N = (c / thick / (2 * np.pi) / (freq * 1e9)) * PhaseD + 1
    T = np.abs(FreqSamP1) / np.abs(FreqRefP1)
    return T, N


def StepCalc(a):
    d = a[1::] - a[0:-1]
    return np.mean(d)


# %%


def prop(RefFilename="",
         SamFilename="",
         FreqUp=0,
         FreqDown=0,
         FreqStep=0,
         TimeSmooth=False,
         FilterSmooth=False,
         SamTime1=2.263,
         SamTime2=2.37745,
         RefTime1=2.263,
         RefTime2=2.37745):

    if RefFilename == "" or SamFilename == "":
        print("Please input the filename")
        return 0

    dataRefI = pre_process(filename=RefFilename, ifInterp=False)
    dataSamI = pre_process(filename=SamFilename, ifInterp=False)

    if (FreqUp - FreqDown) == 0:
        FreqUp = min(np.amax(dataRefI[:, 0]), np.amax(dataSamI[:, 0]))
        FreqDown = max(np.amin(dataRefI[:, 0]), np.amin(dataSamI[:, 0]))

    if FreqStep == 0:
        FreqStep = StepCalc(dataRefI[:, 0])
    StandardFreq = np.arange(FreqDown, FreqUp, FreqStep)
    print("FreqDown FreqUp FreqStep=", FreqDown, FreqUp, FreqStep)
    print("TimeSmooth=",TimeSmooth,"FilterSmooth",FilterSmooth)
    dataRefI = pre_process(filename=RefFilename,
                           ifInterp=True,
                           Freq=StandardFreq)
    dataSamI = pre_process(filename=SamFilename,
                           ifInterp=True,
                           Freq=StandardFreq)

    plt.figure()
    plt.plot(dataRefI[:, 0], dataRefI[:, 1])
    plt.plot(dataSamI[:, 0], dataSamI[:, 1])

    HerRef1, HerRef2 = MHilbert(dataRefI)
    HerSam1, HerSam2 = MHilbert(dataSamI)

    plt.figure()
    plt.plot(StandardFreq, np.abs(HerRef1))
    plt.figure()
    plt.plot(StandardFreq, np.abs(HerRef1))
    plt.plot(StandardFreq, np.abs(HerSam1))

    t, TimeRef = Freq2Time(HerRef1, dFreq=FreqStep, method="IRFFT")
    t, TimeSam = Freq2Time(HerSam1, dFreq=FreqStep, method="IRFFT")
    TimeSave = np.column_stack((t, TimeRef, TimeSam))
    np.savetxt(SamFilename[0:-4] + "-TimeSave.txt", TimeSave)

    plt.figure()
    plt.plot(t, TimeRef)
    plt.plot(t, TimeSam)
    plt.axvline(RefTime1, 0, 1, color="b")
    plt.axvline(RefTime2, 0, 1, color="g")
    plt.axvline(SamTime1, 0, 1, color="r", linestyle="--")
    plt.axvline(SamTime2, 0, 1, color="y", linestyle="--")
    plt.xlim(min(RefTime1, SamTime1) - 0.2, max(RefTime1, SamTime1) + 0.2)

    if TimeSmooth:
        SpecRefS, temp = Smooth_TimeDomain(HilbertResult=HerRef1,
                                           Down=RefTime1,
                                           Up=RefTime2,
                                           dFreq=FreqStep)
        SpecSamS, temp = Smooth_TimeDomain(HilbertResult=HerSam1,
                                           Down=SamTime1,
                                           Up=SamTime2,
                                           dFreq=FreqStep)
    else:
        SpecRefS = HerRef1
        SpecSamS = HerSam1

    Both_Smooth_Amp_Hilbert = np.column_stack(
        (StandardFreq, abs(SpecRefS), abs(SpecSamS)))
    np.savetxt(SamFilename[0:-4] + "-Both_Smooth_Amp_Hilbert.txt",
               Both_Smooth_Amp_Hilbert)

    plt.figure()
    plt.plot(dataRefI[:, 0], np.abs(SpecRefS))
    plt.plot(dataSamI[:, 0], np.abs(SpecSamS))

    T, N = OpticalValue(freq=StandardFreq,
                        FreqSamP1=SpecSamS,
                        FreqRefP1=SpecRefS,
                        DD=6 * np.pi,
                        thick=0.846)
    plt.figure(99)
    plt.plot(StandardFreq, T)
    Smooth_T = np.column_stack((StandardFreq, T))
    np.savetxt(SamFilename[0:-4] + "-Smooth_T.txt", Smooth_T)
    plt.ylim((0, 2))
    plt.figure()
    plt.scatter(StandardFreq, N)
    plt.ylim((0, 2))
    plt.show()
    # tt,NT=Freq2Time(HerSam1/HerRef1, dFreq=FreqStep, method="IRFFT")
    # plt.figure()
    # plt.plot(tt,NT)


# %%
plt.close("all")
if __name__ == "__main__":
    prop(RefFilename="20190820_003-N2.txt",
         SamFilename="20190820_001-water vapour.txt",
         FreqUp=0,
         FreqDown=0,
         FreqStep=0,
         TimeSmooth=True,
         FilterSmooth=False,
         SamTime1=2.257,
         SamTime2=2.375,
         RefTime1=2.257,
         RefTime2=2.375)

# %%
'''
from sympy.core.symbol import *
from sympy import *

T,R,a,ep1,ep2,w,n,k=symbols('T,R,a,ep1,ep2,w,n,k')


R=((n-1)**2+k**2)/((n+1)**2+k**2)
a=2*k*w/c
f=((1-R)**2*E**(-a*d))/((1-R*E**(-a*d))**2+4*R*E**(-a*d)*(sin(n*w*d/c))**2)-T
print(f)
'''
'''
import gc
del Amp_Ref
del Amp_Sam
del Ang_Sam
del Ang_Ref
del Both_Complex1_Hilbert
del Both_Complex2_Hilbert
del Cpx_Ref_Data
del Cpx_Sam_Data
del FreqRefP1
del FreqSamP1
del Her_Ref
del Her_Ref2
del Her_Sam
del Her_Sam2
del PhaseD
del Ref_Fact
del Ref_I
del Sam_Fact
del Sam_I
del ref_data
del ref_data_num
del ref_filename
del ref_inter
del sam_data
del sam_data_num
del sam_filename
del sam_inter
gc.collect()
'''
'''
k0=-np.log(Transmittance*(N+1)**2/(4*N))*c/(2*np.pi*Freq*(10**9)*d)

def f(k,*args): #k消光系数  arg[0]=T,arg[1]=w,arg[2]=n
    return -args[0] + (-(k**2 + (args[2] - 1)**2)/(k**2 + (args[2] + 1)**2) + 1)**2*np.exp(-3.48240915386871e-12*k*args[1])/(4*(k**2 + (args[2] - 1)**2)*np.exp(-3.48240915386871e-12*k*args[1])*np.sin(1.74120457693435e-12*args[2]*args[1])**2/(k**2 + (args[2] + 1)**2) + (-(k**2 + (args[2] - 1)**2)*np.exp(-3.48240915386871e-12*k*args[1])/(k**2 + (args[2]+ 1)**2) + 1)**2)



k=np.zeros(len(Freq))
    
plt.show()

for i in range(len(Freq)):
    try:
        aaa=(Transmittance[i]**2,2*np.pi*Freq[i]*10**9,N[i])
        k[i]=fsolve(f,x0=k0[i],args=aaa[0:],maxfev=1000,xtol=1e-10)
    except:
        k[i]=100
    print("is",i+1,"/",len(Freq))   
    
eps1=N**2-k**2    
eps2=2*N*k
plt.figure(4)    
plt.plot(Freq,eps1)
plt.plot(Freq,eps2)    
plt.ylim(0,5)

'''
'''
from joblib import Parallel, delayed
import multiprocessing
    
# what are your inputs, and what operation do you want to 
# perform on each input. For example...
inputs = range(10) 
def processInput(i):
    try:
        aaa=(Transmittance[i],2*np.pi*Freq[i]*10**9,N[i])
        k[i]=fsolve(f,x0=k0[i],args=aaa[0:])
    except:
        k[i]=100
    print("is",i)   

num_cores = multiprocessing.cpu_count()
    
results = Parallel(n_jobs=num_cores-3)(delayed(processInput)(i) for i in range(len(Freq)))

'''

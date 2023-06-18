import sys
from time import sleep

import librosa
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm

# 自定义字体文件路径
font_path = r'C:\Windows\Fonts\simhei.ttf'
# 加载字体文件
prop = fm.FontProperties(fname=font_path)


def enframe(x, win, inc=None):
    nx = len(x)

    if isinstance(win, (list, np.ndarray)):
        nwin = len(win)
        nlen = nwin  # 帧长=窗长
    elif isinstance(win, int):
        nwin = 1
        nlen = win  # 设置为帧长

    if inc is None:
        inc = nlen

    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))

    indf = inc * np.array(range(nf))

    for i in range(nf):
        frameout[i, :] = x[indf[i]:indf[i] + nlen]

    if isinstance(win, (list, np.ndarray)):
        frameout = frameout * np.array(win)

    return frameout


def calculate_frame_center_times(num_frames, frame_length, frame_shift, sample_rate):
    # 创建长度为 num_frames 的一维数组，元素分别为 0 到 num_frames-1
    frame_indices = np.arange(num_frames)
    # 计算每个帧的中心时间
    frame_center = (frame_indices * frame_shift + frame_length / 2) / sample_rate
    # 返回每帧的中心时间
    return frame_center


def STEn(x, win, inc):
    """
    计算短时能量函数
    :param x: 时域信号
    :param win: 窗函数
    :param inc: 帧移
    :return: 短时能量
    """
    frames = enframe(x, win, inc)
    frames_squared = np.square(frames)
    energy = np.sum(frames_squared, axis=1)
    return energy


def STZcr(x, win, inc, delta=0):
    """
    计算短时过零率
    :param x: 时域信号
    :param win: 窗函数
    :param inc: 帧移
    :param delta: 阈值
    :return: 短时过零率
    """
    absx = np.abs(x)
    x = np.where(absx < delta, 0, x)
    frames = enframe(x, win, inc)
    frames_left = frames[:, :-1]
    frames_right = frames[:, 1:]
    product = np.multiply(frames_left, frames_right)
    sign = np.where(product < 0, 1, 0)
    zcr = np.sum(sign, axis=1)
    return zcr


def findSegment(binary_vector):
    """
    分割成语音段
    :param binary_vector: 表示语音与非语音的二元向量
    :return: 包含语音段信息的字典
    """
    if len(binary_vector) == 0:
        return dict()
    # 获取语音信号的位置
    if binary_vector[0] == 0:
        voiceIndex = np.where(binary_vector)
    else:
        voiceIndex = binary_vector

    # 计算相邻语音信号距离大于1的索引值
    d_voice = np.where(np.diff(voiceIndex) > 1)[0]

    voiceseg = {}
    if len(d_voice) > 0:
        for i in range(len(d_voice) + 1):
            seg = {}
            if i == 0:
                st = voiceIndex[0]
                en = voiceIndex[d_voice[i]]
            elif i == len(d_voice):
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[-1]
            else:
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[d_voice[i]]

            seg['start'] = st
            seg['end'] = en
            seg['duration'] = en - st + 1
            voiceseg[i] = seg

    return voiceseg


def voice_activity_detection_thresholding(x, wlen, inc, NIS):
    """
    使用门限法检测语音段
    :param x: 语音信号
    :param wlen: 分帧长度
    :param inc: 帧移
    :param NIS: 无信号数据
    :return:
    """
    max_silence_frames = 15
    min_utterance_frames = 5

    status = 0

    # 对语音信号进行分帧并逐帧检测
    frames = enframe(x, wlen, inc)
    num_frames = frames.shape[0]

    amp = STEn(x, wlen, inc)
    zcr = STZcr(x, wlen, inc, delta=0.01)

    # 设置阈值
    amp_threshold = np.mean(amp[:NIS])
    zcr_threshold = np.mean(zcr[:NIS])
    amp2_threshold = 2 * amp_threshold
    amp1_threshold = 4 * amp_threshold
    zcr2_threshold = 2 * zcr_threshold

    # 初始化一些变量
    utterance_count = np.zeros(num_frames)
    silence_count = np.zeros(num_frames)
    start_frame_index = np.zeros(num_frames)
    end_frame_index = np.zeros(num_frames)
    current_utterance_index = 0

    for i in range(num_frames):
        if status == 0 or status == 1:
            # 判断当前帧是否为语音帧
            if amp[i] > amp1_threshold:
                start_frame_index[current_utterance_index] = max(1, i - utterance_count[current_utterance_index] - 1)
                status = 2
                silence_count[current_utterance_index] = 0
                utterance_count[current_utterance_index] += 1
            # 判断当前帧是否为可能的语音帧
            elif amp[i] > amp2_threshold or zcr[i] > zcr2_threshold:
                status = 1
                utterance_count[current_utterance_index] += 1
            else:
                status = 0
                utterance_count[current_utterance_index] = 0
                start_frame_index[current_utterance_index] = 0
                end_frame_index[current_utterance_index] = 0

        elif status == 2:
            # 判断当前帧是否为非语音帧
            if amp[i] > amp2_threshold and zcr[i] > zcr2_threshold:
                utterance_count[current_utterance_index] += 1
            else:
                silence_count[current_utterance_index] += 1
                if silence_count[current_utterance_index] < max_silence_frames:
                    utterance_count[current_utterance_index] += 1
                elif utterance_count[current_utterance_index] < min_utterance_frames:
                    status = 0
                    silence_count[current_utterance_index] = 0
                    utterance_count[current_utterance_index] = 0
                else:
                    status = 3
                    end_frame_index[current_utterance_index] = \
                        start_frame_index[current_utterance_index] + utterance_count[current_utterance_index]

        elif status == 3:
            # 当一次检测完成后，更新索引和变量
            status = 0
            current_utterance_index += 1
            utterance_count[current_utterance_index] = 0
            silence_count[current_utterance_index] = 0
            start_frame_index[current_utterance_index] = 0
            end_frame_index[current_utterance_index] = 0

    # 获取语音/非语音序列及其长度
    SF = np.zeros(num_frames)
    NF = np.ones(num_frames)
    for i in range(current_utterance_index):
        SF[int(start_frame_index[i]):int(end_frame_index[i])] = 1
        NF[int(start_frame_index[i]):int(end_frame_index[i])] = 0

    voiceseg = findSegment(np.where(SF == 1)[0])
    num_voicesegs = 0
    if voiceseg:
        num_voicesegs = len(voiceseg.keys())

    # 返回语音段信息、语音段数量、语音/非语音序列、能量值和过零率
    return voiceseg, num_voicesegs, SF, NF, amp, zcr


def voice_seg(data, sr):
    data = data.astype(np.float64)
    # 归一化处理
    data /= np.max(data)
    N = len(data)
    wlen = 200
    inc = 80
    IS = 0.1
    overlap = wlen - inc
    NIS = int((IS * sr - wlen) // inc + 1)
    fn = (N - wlen) // inc + 1

    frameTime = calculate_frame_center_times(fn, wlen, inc, sr)
    time = [i / sr for i in range(N)]

    voiceseg, vsl, SF, NF, amp, zcr = voice_activity_detection_thresholding(data, wlen, inc, NIS)

    return time, data, frameTime, amp, zcr, vsl, voiceseg


def awgn(x, snr, out='signal', method='vectorized', axis=0):
    # 计算信号的功率
    if method == 'vectorized':
        N = x.size
        Ps = np.sum(x ** 2 / N)

    elif method == 'max_en':
        N = x.shape[axis]
        Ps = np.max(np.sum(x ** 2 / N, axis=axis))

    elif method == 'axial':
        N = x.shape[axis]
        Ps = np.sum(x ** 2 / N, axis=axis)

    else:
        raise ValueError('method \"' + str(method) + '\" not recognized.')

    # 将信号功率转换为 dB 单位
    Psdb = 10 * np.log10(Ps)

    # 计算所需的噪声水平
    Pn = Psdb - snr

    # 生成高斯噪声向量（或矩阵）
    n = np.sqrt(10 ** (Pn / 10)) * np.random.normal(0, 1, x.shape)

    # 根据 out 参数选择返回结果
    if out == 'signal':
        return x + n

    elif out == 'noise':
        return n

    elif out == 'both':
        return x + n, n

    else:
        return x + n

# def awgn(x, snr):
#     snr = 10 ** (snr / 10.0)
#     xpower = np.sum(x ** 2) / len(x)
#     npower = xpower / snr
#     return x + np.random.randn(len(x)) * np.sqrt(npower)


def noise_audio(data, snr=10):

    return awgn(data, snr)


def denoise_audio(data, sr):
    data = data.astype(np.float64)
    # 计算 nosiy 信号的频谱
    S_noisy = librosa.stft(data, n_fft=256, hop_length=128, win_length=256)  # D x T
    D, T = np.shape(S_noisy)
    Mag_noisy = np.abs(S_noisy)
    Phase_nosiy = np.angle(S_noisy)
    Power_nosiy = Mag_noisy ** 2
    # 估计噪声信号的能量
    # 由于噪声信号未知 这里假设 含噪（noisy）信号的前30帧为噪声
    Mag_nosie = np.mean(np.abs(S_noisy[:, :31]), axis=1, keepdims=True)
    Power_nosie = Mag_nosie ** 2
    Power_nosie = np.tile(Power_nosie, [1, T])

    Mag_noisy_new = np.copy(Mag_noisy)
    k = 1
    for t in range(k, T - k):
        Mag_noisy_new[:, t] = np.mean(Mag_noisy[:, t - k:t + k + 1], axis=1)

    Power_nosiy = Mag_noisy_new ** 2

    # 超减法去噪
    alpha = 4
    gamma = 1

    Power_enhenc = np.power(Power_nosiy, gamma) - alpha * np.power(Power_nosie, gamma)
    Power_enhenc = np.power(Power_enhenc, 1 / gamma)

    # 对于过小的值用 beta* Power_nosie 替代
    beta = 0.0001
    mask = (Power_enhenc >= beta * Power_nosie) - 0
    Power_enhenc = mask * Power_enhenc + beta * (1 - mask) * Power_nosie

    Mag_enhenc = np.sqrt(Power_enhenc)

    Mag_enhenc_new = np.copy(Mag_enhenc)
    # 计算最大噪声残差
    maxnr = np.max(np.abs(S_noisy[:, :31]) - Mag_nosie, axis=1)

    k = 1
    for t in range(k, T - k):
        index = np.where(Mag_enhenc[:, t] < maxnr)[0]
        temp = np.min(Mag_enhenc[:, t - k:t + k + 1], axis=1)
        Mag_enhenc_new[index, t] = temp[index]

    # 对信号进行恢复
    S_enhec = Mag_enhenc_new * np.exp(1j * Phase_nosiy)
    enhenc = librosa.istft(S_enhec, hop_length=128, win_length=256)

    return enhenc

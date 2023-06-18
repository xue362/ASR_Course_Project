import logging
import os
import sys
import threading
import warnings
import wave
import numpy as np
import pyaudio
import sounddevice
import librosa
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QFileDialog, QLabel
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from SR_CMU_Sphinx import SR_Sphinx
from SR_Conformer import SR_CF
from util import prop, voice_seg, noise_audio, denoise_audio

import ctypes
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"


def disable_console_output():
    # 禁用 stdout 和 stderr
    sys.stdout = open('nul', 'w')
    sys.stderr = open('nul', 'w')
    # 禁用警告信息
    warnings.filterwarnings("ignore")
    # 禁用日志信息
    logging.disable(logging.CRITICAL)


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.playing = None
        self.recording = False
        self.frames = None
        self.channels = None
        self.sample_width = None
        self.original_audio = None
        self.audio_name = None
        self.sample_rate = None
        self.current_audio = None
        self.noised_audio_data = None
        self.denoised_audio = None
        self.audio_duration = None

        self.setWindowTitle('语音识别演示')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(f"background-color: {format(QColor('#E6E6FA').name())}")

        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)

        button_size = (100, 40)
        self.buttons = []
        for i in range(1, 10):
            button = QPushButton(f'Button {i}')
            button.setFixedSize(*button_size)
            button.setStyleSheet("background-color: {0}; border-radius: 6px".format(QColor("#ffffff").name()))
            self.buttons.append(button)

        button_layout = QVBoxLayout()
        for button in self.buttons:
            button_layout.addWidget(button)

        self.tip_label = QLabel()
        self.tip_label.setFixedHeight(40)
        self.tip_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.tip_label.setStyleSheet("""
                    border: 1px solid gray;
                    border-radius: 6px;
                    padding: 8px;
                    background: #ffffff;
                """)
        font = QFont()
        font.setPointSize(12)
        self.tip_label.setFont(font)
        self.update_tip_label("请打开音频文件或使用录制功能")

        fig_layout = QVBoxLayout()
        fig_layout.addWidget(self.canvas)
        fig_layout.addWidget(self.tip_label)

        layout = QHBoxLayout()
        layout.addLayout(button_layout)
        layout.addLayout(fig_layout)

        layout.setContentsMargins(50, 50, 50, 50)

        self.setLayout(layout)

        self.buttons[8].setText("播放音频")
        self.buttons[8].clicked.connect(self.play_audio)
        self.buttons[8].setStyleSheet(
            "border-radius: 6px;"
            "color: {0}; background-color: {1}".format(QColor("#8B4513").name(), QColor("#ffffff").name()))
        self.buttons[1].setText("打开文件")
        self.buttons[1].setStyleSheet(
            "border-radius: 6px;"
            "color: {0}; background-color: {1}".format(QColor("#191970").name(), QColor("#ffffff").name()))
        self.buttons[1].clicked.connect(self.choose_file)
        self.buttons[2].setText("添加噪声")
        self.buttons[2].clicked.connect(self.noise_audio_plot)
        self.buttons[3].setText("语音降噪/增强")
        self.buttons[3].clicked.connect(self.denoise_audio_plot)
        self.buttons[4].setText("端点检测")
        self.buttons[4].clicked.connect(self.voice_seg_plot)
        self.buttons[5].setText("音频特征")
        self.buttons[5].clicked.connect(self.MFCC)
        button_font = QFont()
        button_font.setPointSize(8)
        self.buttons[6].setFont(button_font)
        self.buttons[7].setFont(button_font)
        self.buttons[6].setText("语音识别\nHMM(Sphinx)")
        self.buttons[6].clicked.connect(self.SR_Sphinx)
        self.buttons[7].setText("语音识别\nConformer")
        self.buttons[7].clicked.connect(self.SR_Conformer)

        self.g = self.record_generator("Record_out.wav", self.buttons[0])
        next(self.g)
        self.buttons[0].clicked.connect(lambda: next(self.g))
        self.buttons[0].setStyleSheet(
            "border-radius: 6px;"
            "color: {0}; background-color: {1}".format(QColor("#800000").name(), QColor("#ffffff").name()))

    def record_thread(self, fileName, stream, p):
        waveFile = wave.open(fileName, 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(44100)
        while self.recording:
            waveFile.writeframes(stream.read(1024))
        waveFile.close()
        self.update_tip_label("音频已被读入并保存到 Record_out.wav", color="blue")
        self.audio_name = "Recorded"
        self.read_waveform_plot(fileName)

    def record_generator(self, fileName, recordBtn):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1, rate=44100,
                        input=True, frames_per_buffer=1024)
        while 1:
            recordBtn.setText('录制音频')
            yield
            recordBtn.setText('停止录制')
            self.update_tip_label("录制音频中...", "red")
            self.recording = True
            t = threading.Thread(target=self.record_thread, args=(fileName, stream, p))
            t.daemon = True
            t.start()
            yield
            self.recording = False

    def update_tip_label(self, content, color="black"):
        self.tip_label.setText(content)
        palette = QPalette()
        palette.setColor(QPalette.WindowText, QColor(color))  # 设置文本颜色为红色
        self.tip_label.setPalette(palette)

    def choose_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "选择语音文件", "", "Wave Files (*.wav);;All Files (*)",
                                                  options=options)
        if filename:
            # shutil.copy2(filename, "./AppData/audio.wav")
            self.audio_name = os.path.basename(filename)
            self.read_waveform_plot(filename)

    def read_waveform_plot(self, filename):
        self.figure.clear()
        # 读取音频文件数据
        try:
            with wave.open(filename, 'rb') as wav_file:
                self.frames = wav_file.readframes(-1)
                self.sample_rate = wav_file.getframerate()
                self.audio_duration = wav_file.getnframes() / self.sample_rate
                self.sample_width = wav_file.getsampwidth()
                self.channels = wav_file.getnchannels()
        except:
            self.update_tip_label("打开音频文件失败", color="red")

        if self.frames is None:
            self.update_tip_label("打开音频文件失败", color="red")
            return
        self.update_tip_label("打开音频文件成功", color="blue")
        # 将字节数组转换为 16 位有符号整数数组
        waveform = np.frombuffer(self.frames, dtype=np.int16)

        self.original_audio = waveform
        self.current_audio = self.original_audio

        # 绘制波形图
        time_axis = np.linspace(0, self.audio_duration, num=len(waveform))
        ax = self.figure.add_subplot(111)
        ax.clear()  # 清空当前的 Axes
        ax.plot(time_axis, waveform)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform of \"{}\"".format(self.audio_name))
        plt.tight_layout()

        # 在 FigureCanvas 对象中绘制 Matplotlib 的 Figure 对象
        self.canvas.draw_idle()

    def MFCC(self):
        if self.frames is None:
            self.update_tip_label("未打开文件", "red")
            return

        D = librosa.stft(self.current_audio.astype(np.float32))
        sr = self.sample_rate
        self.figure.clear()
        ax = self.figure.add_subplot(211)
        bx = self.figure.add_subplot(212)

        ax.set_title('Linear-frequency power spectrogram')
        img=librosa.display.specshow(librosa.amplitude_to_db(abs(D), ref=np.max),
                                 y_axis='linear', x_axis='time',ax=ax)
        plt.colorbar(img, ax=ax, format="%+2.f dB")
        ax.set_xlabel("Time(s)")

        bx.set_title('MFCC')
        mfccs = librosa.feature.mfcc(y=self.current_audio.astype(np.float32), sr=sr, n_mfcc=20)
        img = librosa.display.specshow(mfccs, x_axis='time',ax=bx,cmap='coolwarm')
        plt.colorbar(img,ax=bx,format="%+2.f dB")
        bx.set_xlabel("Time(s)")
        plt.tight_layout()

        self.canvas.draw_idle()

    def voice_seg_plot(self):
        if self.frames is None:
            self.update_tip_label("未打开文件", "red")
            return

        time, data, frameTime, amp, zcr, vsl, voiceseg = voice_seg(self.current_audio, self.sample_rate)
        self.figure.clear()
        ax = self.figure.add_subplot(311)
        bx = self.figure.add_subplot(312)
        cx = self.figure.add_subplot(313)

        ax.set_title("语音片段分割 - 端点识别", fontdict=dict(fontfamily=prop.get_name()))

        ax.plot(time, data)
        ax.set_ylabel('波形图', fontdict=dict(fontfamily=prop.get_name()))
        ax.set_xlabel('时间(s)', fontdict=dict(fontfamily=prop.get_name()))

        bx.plot(frameTime, amp)
        bx.set_ylabel('短时能量', fontdict=dict(fontfamily=prop.get_name()))
        bx.set_xlabel('时间(s)', fontdict=dict(fontfamily=prop.get_name()))

        cx.plot(frameTime, zcr)
        cx.set_ylabel('过零率', fontdict=dict(fontfamily=prop.get_name()))
        cx.set_xlabel('时间(s)', fontdict=dict(fontfamily=prop.get_name()))

        if vsl == 0:
            self.update_tip_label("未找到显著语音端点", color="red")
        else:
            self.update_tip_label(f"找到{vsl}个语音片段", color="blue")
        for i in range(vsl):
            ax.axvline(x=frameTime[voiceseg[i]['start']], color='red', linestyle='--')
            ax.axvline(x=frameTime[voiceseg[i]['end']], color='black', linestyle='-')
            bx.axvline(x=frameTime[voiceseg[i]['start']], color='red', linestyle='--')
            bx.axvline(x=frameTime[voiceseg[i]['end']], color='black', linestyle='-')
            cx.axvline(x=frameTime[voiceseg[i]['start']], color='red', linestyle='--')
            plt.axvline(x=frameTime[voiceseg[i]['end']], color='black', linestyle='-')
        plt.tight_layout()
        self.canvas.draw_idle()

    def noise_audio_plot(self):
        if self.frames is None:
            self.update_tip_label("未打开文件", "red")
            return

        self.figure.clear()
        self.noised_audio_data = noise_audio(self.current_audio)
        self.current_audio = self.noised_audio_data

        time_axis = np.linspace(0, self.audio_duration, num=len(self.noised_audio_data))
        ax = self.figure.add_subplot(111)
        ax.clear()  # 清空当前的 Axes
        ax.plot(time_axis, self.noised_audio_data)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform of noised \"{}\"".format(self.audio_name))
        plt.tight_layout()
        self.canvas.draw_idle()

    def denoise_audio_plot(self):
        if self.frames is None:
            self.update_tip_label("未打开文件", "red")
            return

        self.figure.clear()
        self.denoised_audio = denoise_audio(self.current_audio, self.sample_rate)
        self.current_audio = self.denoised_audio

        time_axis = np.linspace(0, self.audio_duration, num=len(self.current_audio))
        ax = self.figure.add_subplot(111)
        ax.clear()  # 清空当前的 Axes
        ax.plot(time_axis, self.denoised_audio)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform of Denoised \"{}\"".format(self.audio_name))
        plt.tight_layout()
        self.canvas.draw_idle()

    def SR_Sphinx(self):
        if self.frames is None:
            self.update_tip_label("未打开文件", "red")
            return
        SR_content = SR_Sphinx.SR(self.frames, self.sample_rate, self.sample_width)
        self.update_tip_label(SR_content, "blue")

    def SR_Conformer(self):
        if self.frames is None:
            self.update_tip_label("未打开文件", "red")
            return
        SR_content = SR_CF.SR(self.current_audio, self.sample_rate)
        self.update_tip_label(SR_content["text"], "blue")

    def play_audio(self):
        if self.frames is None:
            self.update_tip_label("未打开文件", "red")
            return
        self.update_tip_label("正在播放音频，请等待", "red")

        def play_audio_thread():
            sounddevice.play(self.current_audio, self.sample_rate)
            sounddevice.wait()
            self.update_tip_label("播放结束，可继续操作", "blue")
            self.playing = False

        thread = threading.Thread(target=play_audio_thread, daemon=True)
        self.playing = True
        thread.start()


if __name__ == '__main__':
    disable_console_output()
    app = QApplication(sys.argv)
    font = QFont()
    font.setFamily('Microsoft YaHei')
    font.setPointSize(10)
    app.setFont(font)
    window = MyWindow()
    icon = QIcon('head.ico')
    window.setWindowIcon(icon)
    app.setWindowIcon(icon)
    window.showMaximized()
    sys.exit(app.exec_())

import speech_recognition as sr

lan_set = \
    ("SR_CMU_Sphinx/Trained_model_ZH_CN/acoustic-model",
     "SR_CMU_Sphinx/Trained_model_ZH_CN/language-model.lm.bin",
     "SR_CMU_Sphinx/Trained_model_ZH_CN/pronounciation-dictionary.dict")


def SR(audio_data, sample_rate, sample_width):
    audio = sr.AudioData(audio_data, sample_rate=sample_rate, sample_width=sample_width)

    r = sr.Recognizer()  # 调用识别器

    return r.recognize_sphinx(audio, language=lan_set)  # 识别输出

import pyaudio
import wave
import numpy as np
import speech_recognition as sr
from scipy.signal import butter, sosfreqz, sosfilt

def butter_bandstop_filter(data, lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(N=order, Wn=[low, high], btype='bandstop', analog=False, output='sos')
    y = sosfilt(sos, data)
    return y.astype(np.int16)

def main():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    frames = []

    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

        frames = b''.join(frames)

        stop_band = (4000, 6000)
        frames_filtered = butter_bandstop_filter(np.frombuffer(frames, dtype=np.int16), stop_band[0], stop_band[1], fs=44100)

        sound_file = wave.open("recording_with_filter.wav", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(frames_filtered.tobytes())
        sound_file.close()


        #speech recognition section
        audio_file = sr.AudioFile("recording_with_filter.wav")
        
        recognizer= sr.Recognizer()
        recognizer.energy_threshold=300

        with audio_file as source:
            
            audio_file_data= recognizer.record(source)
        text= recognizer.recognize_google(audio_data=audio_file_data, language="en-US")
        print(text)

if __name__ == "__main__":
    main()

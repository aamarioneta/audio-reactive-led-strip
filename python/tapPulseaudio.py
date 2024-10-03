import pyaudio
import numpy as np
import time

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

SPEAKERS = p.get_default_output_device_info()["hostApi"] #The part I have modified

def callback(in_data, frame_count, time_info, status):
    audio_data = np.fromstring(in_data, dtype=np.float32)
    print(audio_data);
    return (in_data, pyaudio.paContinue)

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_host_api_specific_stream_info=SPEAKERS,
                stream_callback=callback
                )
while stream.is_active():
        time.sleep(0.1)

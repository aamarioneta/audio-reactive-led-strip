import time
import numpy as np
import pyaudio
import config
import sys
import soundfile as sf

p = pyaudio.PyAudio()
SPEAKERS = p.get_default_output_device_info()["hostApi"]

def start_stream(callback):
    print("start_stream stdin");
    frames_per_buffer = int(config.MIC_RATE / config.FPS)
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=config.MIC_RATE,
                    input=True,
                    frames_per_buffer=frames_per_buffer,
                    input_host_api_specific_stream_info=SPEAKERS)
    overflows = 0
    prev_ovf_time = time.time()
    while True:
        try:
            y = np.fromstring(stream.read(frames_per_buffer, exception_on_overflow=False), dtype=np.int16)
            y = y.astype(np.float32)
            #if config.DISPLAY_FPS:
            #    print(y)
            callback(y)
        except IOError:
            overflows += 1
            if time.time() > prev_ovf_time + 1:
                prev_ovf_time = time.time()
                print('Audio buffer has overflowed {} times'.format(overflows))
    stream.stop_stream()
    stream.close()
    p.terminate()
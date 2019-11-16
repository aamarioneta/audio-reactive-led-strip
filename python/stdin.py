import time
import numpy as np
import pyaudio
import config
import sys
import soundfile as sf

def start_stream(callback):
    frames_per_buffer = int(config.MIC_RATE / config.FPS)
    overflows = 0
    prev_ovf_time = time.time()
    while True:
        try:
            data = np.fromstring(sys.stdin.buffer.read(frames_per_buffer * 8),dtype=int)
            callback(data)
        except IOError:
            overflows += 1
            if time.time() > prev_ovf_time + 1:
                prev_ovf_time = time.time()
                print('Audio buffer has overflowed {} times'.format(overflows))

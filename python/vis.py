from __future__ import print_function
from __future__ import division
import json
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import config
import microphone
import stdin
import dsp
import led
import threading
from flask import Flask
from flask import render_template
from flask import request
import socket
import struct
import sys
import time

max_brightness = 64.0
CURRENT_VISUALIZATION = "visualize_vumeter"
current_visualization_name = "visualize_amplitudePerFrequency"

_time_prev = time.time() * 1000.0
"""The previous time that the frames_per_second() function was called"""

_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)
"""The low-pass filter used to estimate frames-per-second"""

app = Flask(__name__)


def frames_per_second():
    """Return the estimated frames per second

    Returns the current estimate for frames-per-second (FPS).
    FPS is estimated by measured the amount of time that has elapsed since
    this function was previously called. The FPS estimate is low-pass filtered
    to reduce noise.

    This function is intended to be called one time for every iteration of
    the program's main loop.

    Returns
    -------
    fps : float
        Estimated frames-per-second. This value is low-pass filtered
        to reduce noise.
    """
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)


def memoize(function):
    """Provides a decorator for memoizing functions"""
    from functools import wraps
    memo = {}

    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv

    return wrapper


@memoize
def _normalized_linspace(size):
    return np.linspace(0, 1, size)


def interpolate(y, new_length):
    """Intelligently resizes the array by linearly interpolating the values

    Parameters
    ----------
    y : np.array
        Array that should be resized

    new_length : int
        The length of the new interpolated array

    Returns
    -------
    z : np.array
        New array with length of new_length that contains the interpolated
        values of y.
    """
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z


r_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.2, alpha_rise=0.99)
g_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.05, alpha_rise=0.3)
b_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.1, alpha_rise=0.5)
common_mode = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                            alpha_decay=0.99, alpha_rise=0.01)
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)),
                       alpha_decay=0.1, alpha_rise=0.99)
p = np.tile(1.0, (3, config.N_PIXELS // 2))
gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS),
                     alpha_decay=0.001, alpha_rise=0.99)


def visualize_scroll(y):
    """Effect that originates in the center and scrolls outwards"""
    global p
    y = y ** 2.0
    gain.update(y)
    y /= gain.value
    y *= max_brightness
    r = int(np.max(y[:len(y) // 3]))
    g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = int(np.max(y[2 * len(y) // 3:]))
    # Scrolling effect window
    p[:, 1:] = p[:, :-1]
    p *= 0.98
    p = gaussian_filter1d(p, sigma=0.2)
    # Create new color originating at the center
    p[0, 0] = r
    p[1, 0] = g
    p[2, 0] = b
    # Update the LED strip
    return np.concatenate((p[:, ::-1], p), axis=1)


def visualize_energy(y):
    """Effect that expands from the center with increasing sound energy"""
    global p
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Scale by the width of the LED strip
    y *= float((config.N_PIXELS // 2) - 1)
    # Map color channels according to energy in the different freq bands
    scale = 0.999
    r = int(np.mean(y[:len(y) // 3] ** scale))
    g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3] ** scale))
    b = int(np.mean(y[2 * len(y) // 3:] ** scale))
    # Assign color to different frequency regions
    p[0, :r] = max_brightness
    p[0, r:] = 0.0
    p[1, :g] = max_brightness
    p[1, g:] = 0.0
    p[2, :b] = max_brightness
    p[2, b:] = 0.0
    p_filt.update(p)
    p = np.round(p_filt.value)
    # Apply substantial blur to smooth the edges
    p[0, :] = gaussian_filter1d(p[0, :], sigma=4.0)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=4.0)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=4.0)
    # Set the new pixel value
    return np.concatenate((p[:, ::-1], p), axis=1)


_prev_spectrum = np.tile(0.01, config.N_PIXELS)


def visualize_spectrum(y):
    """Effect that maps the Mel filterbank frequencies onto the LED strip"""
    global _prev_spectrum
    y = np.copy(interpolate(y, config.N_PIXELS // 2))
    _prev_spectrum = np.copy(interpolate(_prev_spectrum, config.N_PIXELS // 2))
    common_mode.update(y)
    try:
        diff = y - _prev_spectrum
        _prev_spectrum = np.copy(y)
        # Color channel mappings
        r = r_filt.update(y - common_mode.value)
        g = np.abs(diff)
        b = b_filt.update(np.copy(y))
        # Mirror the color channels for symmetric output
        r = np.concatenate((r[::-1], r))
        g = np.concatenate((g[::-1], g))
        b = np.concatenate((b[::-1], b))
        output = np.array([r, g, b]) * max_brightness
    except Exception as e:
        print(e)
        r = np.tile(32, config.N_PIXELS)
        g = np.tile(16, config.N_PIXELS)
        b = np.tile(8, config.N_PIXELS)
        return np.array([r, g, b])
    return output


def visualize_amplitude_per_frequency(y):
    global _prev_spectrum
    y = np.copy(interpolate(y, config.N_PIXELS))
    _prev_spectrum = np.copy(y)
    # Color channel mappings
    r = np.tile(0.0001, config.N_PIXELS)
    g = np.tile(0.0001, config.N_PIXELS)
    b = np.tile(0.0001, config.N_PIXELS)
    for i in range(config.N_PIXELS):
        if y[i] < 2 * config.ONE3RD:
            b[i] = y[i]
        else:
            b[i] = 0
        if y[i] > 2 * config.ONE3RD:
            g[i] = y[i]
        else:
            g[i] = 0
        if y[i] > config.ONE3RD:
            r[i] = y[i]
        else:
            r[i] = 0
    output = np.array([r, g, b]) * max_brightness
    return output


def visualize_amplitude_per_frequency_one_color(y):
    global _prev_spectrum
    y = np.copy(interpolate(y, config.N_PIXELS))
    _prev_spectrum = np.copy(y)
    # Color channel mappings
    r = np.tile(0.0001, config.N_PIXELS)
    g = np.tile(0.0001, config.N_PIXELS)
    b = np.tile(0.0001, config.N_PIXELS)
    for i in range(config.N_PIXELS):
        r[i] = y[i]
        b[i] = y[i]
    output = np.array([r, g, b]) * max_brightness
    return output


globalMaxAmp = 0.0000001

globalVUValues = []


def visualize_vumeter(y):
    global _prev_spectrum
    global globalMaxAmp
    y = np.copy(interpolate(y, config.N_PIXELS))
    _prev_spectrum = np.copy(y)
    # Color channel mappings
    r = np.tile(0.0, config.N_PIXELS)
    g = np.tile(0.0, config.N_PIXELS)
    b = np.tile(0.0, config.N_PIXELS)
    avg_amplitude = np.average(y)
    globalMaxAmp = max(avg_amplitude, globalMaxAmp)
    j = int(avg_amplitude * config.N_PIXELS / globalMaxAmp)
    for i in range(config.N_PIXELS):
        if j > i:
            if i < config.N_PIXELS / 3 * 2:
                # first two thirds of the leds are green
                g[i] = 0.1
            else:
                # the last third of the leds are red
                r[i] = 0.1
        else:
            r[i] = 0
            g[i] = 0
    output = np.array([r, g, b]) * max_brightness
    # globalMaxAmp = globalMaxAmp - .001
    vuMeterAmplitude = avg_amplitude * 255 / globalMaxAmp
    # print ("globalMaxAmp: ", globalMaxAmp , "avg_amplitude: ", avg_amplitude, "vuMeterAmplitude: ", vuMeterAmplitude)
    led.globalVUValues = [int(vuMeterAmplitude),int(vuMeterAmplitude)]
    return output


fft_plot_filter = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                                alpha_decay=0.5, alpha_rise=0.99)
mel_gain = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.01, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                              alpha_decay=0.5, alpha_rise=0.99)
volume = dsp.ExpFilter(config.MIN_VOLUME_THRESHOLD,
                       alpha_decay=0.02, alpha_rise=0.02)
fft_window = np.hamming(int(config.MIC_RATE / config.FPS) * config.N_ROLLING_HISTORY)
prev_fps_update = time.time()


def microphone_update(audio_samples):
    global y_roll, prev_rms, prev_exp, prev_fps_update
    # Normalize samples between 0 and 1
    y = audio_samples / config.NR_OF_CHANNELS ** 15
    # Construct a rolling window of audio samples
    y_roll[:-1] = y_roll[1:]
    y_roll[-1, :] = np.copy(y)
    y_data = np.concatenate(y_roll, axis=0).astype(np.float32)

    vol = np.max(np.abs(y_data))
    if vol < config.MIN_VOLUME_THRESHOLD:
        # print('No audio input. Volume below threshold. Volume:', vol)
        led.pixels = np.tile(0.0, (3, config.N_PIXELS))
        # print(led.pixels)
        led.update()
    else:
        # Transform audio input into the frequency domain
        data_length = len(y_data)
        n_zeros = 2 ** int(np.ceil(np.log2(data_length))) - data_length
        # Pad with zeros until the next power of two
        y_data *= fft_window
        y_padded = np.pad(y_data, (0, n_zeros), mode='constant')
        ys = np.abs(np.fft.rfft(y_padded)[:data_length // 2])
        # Construct a Mel filterbank from the FFT data
        mel = np.atleast_2d(ys).T * dsp.mel_y.T
        # Scale data to values more suitable for visualization
        # mel = np.sum(mel, axis=0)
        mel = np.sum(mel, axis=0)
        mel = mel ** 2.0
        # Gain normalization
        mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
        mel /= mel_gain.value
        mel = mel_smoothing.update(mel)
        # Map filterbank output onto LED strip

        #print('--------')
        #led.globalVUValues = np.round_(np.tile((np.average(mel) / 1000)*1000,2), decimals = 3) # np.amax(np.split(y_data,2), axis=1, keepdims=True)
        #print(led.globalVUValues)
        #print('++++++++')
        output = visualization_effect(mel)
        led.pixels = output
        led.update()
        if config.USE_GUI:
            # Plot filterbank output
            x = np.linspace(config.MIN_FREQUENCY, config.MAX_FREQUENCY, len(mel))
            mel_curve.setData(x=x, y=fft_plot_filter.update(mel))
            # Plot the color channels
            r_curve.setData(y=led.pixels[0])
            g_curve.setData(y=led.pixels[1])
            b_curve.setData(y=led.pixels[2])
    if config.USE_GUI:
        app.processEvents()

    if config.DISPLAY_FPS:
        fps = frames_per_second()
        if time.time() - 0.5 > prev_fps_update:
            prev_fps_update = time.time()
            print('FPS {:.0f} / {:.0f}'.format(fps, config.FPS))


# Number of audio samples to read every time frame
samples_per_frame = int(config.MIC_RATE / config.FPS)

# Array containing the rolling audio sample window
y_roll = np.random.rand(config.N_ROLLING_HISTORY, samples_per_frame) / 1e16

visualization_effect = visualize_vumeter
"""Visualization effect to display on the LED strip"""


@app.route("/", methods=['GET', 'POST'])
@app.route("/vis", methods=['GET', 'POST'])
def vis():
    global visualization_effect
    if request.method == 'POST':
        form_vis = request.form["name"]
        if form_vis == "visualize_spectrum":
            visualization_effect = visualize_spectrum
        if form_vis == "visualize_energy":
            visualization_effect = visualize_energy
        if form_vis == "visualize_scroll":
            visualization_effect = visualize_scroll
        if form_vis == "visualize_vumeter":
            visualization_effect = visualize_vumeter
        if form_vis == "visualize_amplitude_per_frequency":
            visualization_effect = visualize_amplitude_per_frequency
        if form_vis == "visualize_amplitude_per_frequency_one_color":
            visualization_effect = visualize_amplitude_per_frequency_one_color
        print(visualization_effect)
    return render_template('index.html', MAX_BRIGHTNESS=max_brightness)


@app.route("/brightness", methods=['POST'])
def brightness():
    global max_brightness
    print(request.form["brightnessValue"])
    max_brightness = int(request.form["brightnessValue"])
    return render_template('index.html', MAX_BRIGHTNESS=max_brightness)


@app.route("/get")
def get():
    return CURRENT_VISUALIZATION


@app.route("/off")
def off():
    led.pixels = np.tile(1.0, (3, config.N_PIXELS))
    led.update()
    led.pixels = np.tile(0.0, (3, config.N_PIXELS))
    led.update()
    return render_template('index.html', MAX_BRIGHTNESS=max_brightness)


@app.route("/config")
def get_config():
    data = {"CURRENT_VISUALIZATION": CURRENT_VISUALIZATION, "MAX_BRIGHTNESS": max_brightness}
    s = json.dumps(data)
    return s


def read_stream():
    print("started read stream thread!!!!!!!!!!!!")
    print("listening to " + config.SOURCE)
    print("UDP", config.UDP_IP, config.UDP_PORT)
    print("ANALOG_VU_METER_IP", config.ANALOG_VU_METER_IP, config.ANALOG_VU_METER_PORT)
    if config.SOURCE == 'stdin':
        stdin.start_stream(microphone_update)
    else:
        microphone.start_stream(microphone_update)

def RequestTimefromNtp(addr='openwrt.lan'):
    REF_TIME_1970 = 2208988800  # Reference time
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = b'\x1b' + 47 * b'\0'
    client.sendto(data, (addr, 123))
    data, address = client.recvfrom(1024)
    if data:
        t = struct.unpack('!12I', data)[10]
        t -= REF_TIME_1970
    return time.ctime(t), t


if __name__ == '__main__':
    thread_read_stream = threading.Thread(target=read_stream)
    thread_read_stream.start()
    if config.USE_GUI:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtGui, QtCore

        # Create GUI window
        app = QtGui.QApplication([])
        view = pg.GraphicsView()
        layout = pg.GraphicsLayout(border=(100, 100, 100))
        view.setCentralItem(layout)
        view.show()
        view.setWindowTitle('Visualization')
        view.resize(800, 600)
        # Mel filterbank plot
        fft_plot = layout.addPlot(title='Filterbank Output', colspan=3)
        fft_plot.setRange(yRange=[-0.1, 1.2])
        fft_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        x_data = np.array(range(1, config.N_FFT_BINS + 1))
        mel_curve = pg.PlotCurveItem()
        mel_curve.setData(x=x_data, y=x_data * 0)
        fft_plot.addItem(mel_curve)
        # Visualization plot
        layout.nextRow()
        led_plot = layout.addPlot(title='Visualization Output', colspan=3)
        led_plot.setRange(yRange=[-5, 260])
        led_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        # Pen for each of the color channel curves
        r_pen = pg.mkPen((255, 30, 30, 200), width=4)
        g_pen = pg.mkPen((30, 255, 30, 200), width=4)
        b_pen = pg.mkPen((30, 30, 255, 200), width=4)
        # Color channel curves
        r_curve = pg.PlotCurveItem(pen=r_pen)
        g_curve = pg.PlotCurveItem(pen=g_pen)
        b_curve = pg.PlotCurveItem(pen=b_pen)
        # Define x data
        x_data = np.array(range(1, config.N_PIXELS + 1))
        r_curve.setData(x=x_data, y=x_data * 0)
        g_curve.setData(x=x_data, y=x_data * 0)
        b_curve.setData(x=x_data, y=x_data * 0)
        # Add curves to plot
        led_plot.addItem(r_curve)
        led_plot.addItem(g_curve)
        led_plot.addItem(b_curve)
        # Frequency range label
        freq_label = pg.LabelItem('')

        # Frequency slider
        def freq_slider_change(tick):
            minf = freq_slider.tickValue(0) ** 2.0 * (config.MIC_RATE / 2.0)
            maxf = freq_slider.tickValue(1) ** 2.0 * (config.MIC_RATE / 2.0)
            t = 'Frequency range: {:.0f} - {:.0f} Hz'.format(minf, maxf)
            freq_label.setText(t)
            config.MIN_FREQUENCY = minf
            config.MAX_FREQUENCY = maxf
            dsp.create_mel_bank()


        freq_slider = pg.TickSliderItem(orientation='bottom', allowAdd=False)
        freq_slider.addTick((config.MIN_FREQUENCY / (config.MIC_RATE / 2.0)) ** 0.5)
        freq_slider.addTick((config.MAX_FREQUENCY / (config.MIC_RATE / 2.0)) ** 0.5)
        freq_slider.tickMoveFinished = freq_slider_change
        freq_label.setText('Frequency range: {} - {} Hz'.format(
            config.MIN_FREQUENCY,
            config.MAX_FREQUENCY))
        # Effect selection
        active_color = '#16dbeb'
        inactive_color = '#FFFFFF'


        def energy_click(x):
            global visualization_effect
            visualization_effect = visualize_vumeter
            energy_label.setText('Energy', color=active_color)
            scroll_label.setText('Scroll', color=inactive_color)
            spectrum_label.setText('Spectrum', color=inactive_color)


        def scroll_click(x):
            global visualization_effect
            visualization_effect = visualize_scroll
            energy_label.setText('Energy', color=inactive_color)
            scroll_label.setText('Scroll', color=active_color)
            spectrum_label.setText('Spectrum', color=inactive_color)


        def spectrum_click(x):
            global visualization_effect
            visualization_effect = visualize_amplitude_per_frequency
            energy_label.setText('Energy', color=inactive_color)
            scroll_label.setText('Scroll', color=inactive_color)
            spectrum_label.setText('Spectrum', color=active_color)


        # Create effect "buttons" (labels with click event)
        energy_label = pg.LabelItem('Energy')
        scroll_label = pg.LabelItem('Scroll')
        spectrum_label = pg.LabelItem('Spectrum')
        energy_label.mousePressEvent = energy_click
        scroll_label.mousePressEvent = scroll_click
        spectrum_label.mousePressEvent = spectrum_click
        spectrum_click(0)
        # Layout
        layout.nextRow()
        layout.addItem(freq_label, colspan=3)
        layout.nextRow()
        layout.addItem(freq_slider, colspan=3)
        layout.nextRow()
        layout.addItem(energy_label)
        layout.addItem(scroll_label)
        layout.addItem(spectrum_label)
    # Initialize LEDs
    led.update()

    # Start listening to live audio stream
    app.run(host='0.0.0.0', port=5001)
    print(RequestTimefromNtp())
    print("started web interface.")

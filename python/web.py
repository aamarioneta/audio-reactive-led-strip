from flask import Flask
from flask import render_template
from flask import request
import json
import led
import config
import numpy as np


app = Flask(__name__)

CURRENT_VISUALIZATION = "visualize_vumeter"
max_brightness = 64


@app.route("/vis", methods=['GET', 'POST'])
def vis():
    global CURRENT_VISUALIZATION, max_brightness
    if request.method == 'POST':
        CURRENT_VISUALIZATION = request.form["name"]
    return render_template('index.html', title='Home', user=CURRENT_VISUALIZATION)


@app.route("/brightness", methods=['POST'])
def brightness():
    global max_brightness
    print(request.form["brightnessValue"])
    max_brightness = int(request.form["brightnessValue"])
    return render_template('index.html', title='Home', user=CURRENT_VISUALIZATION)


@app.route("/get")
def get():
    return CURRENT_VISUALIZATION


@app.route("/off")
def off():
    led.pixels = np.tile(0.0, (3, 54))
    led.update()
    return render_template('index.html', title='Home', user=CURRENT_VISUALIZATION)


@app.route("/config")
def config():
    data = {"CURRENT_VISUALIZATION": CURRENT_VISUALIZATION, "MAX_BRIGHTNESS": max_brightness}
    s = json.dumps(data)
    return s


if __name__ == '__main__':
    app.run(host='0.0.0.0')

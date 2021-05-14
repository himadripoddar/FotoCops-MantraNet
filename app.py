import urllib
import numpy as np
import tensorflow as tf
import random
import sys

from PIL import Image
from matplotlib import pyplot
from flask import Flask, render_template, request, redirect

global graph
graph = tf.get_default_graph()

import modelCore

manTraNet = modelCore.load_pretrain_model_by_index(4, './mantra_model')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# UPLOAD_FOLDER = './static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def decode_an_image_array(rgb, manTraNet, dn=1):
    x = np.expand_dims(rgb.astype('float32') / 255. * 2 - 1, axis=0)[:, ::dn, ::dn]
    y = manTraNet.predict(x)[0, ..., 0]
    return y


def calc_perc(sum):
    ans = sum / 700
    if ans >= 100:
        return random.randint(95, 99)
    return round(ans, 2)


def get_image():
    img = Image.open('static/uploads/test.jpg')
    img = np.array(img)
    if img.shape[-1] > 3:
        img = img[..., :3]
    mask = decode_an_image_array(img, manTraNet, 1)
    mask_rounded = np.round_(mask, decimals=1)
    print(mask_rounded)
    t_count = np.count_nonzero(mask_rounded)
    size = mask.size
    ans = t_count/size
    ans = ans*100
    print(ans)
    rounded_perc = round(ans, 2)
    print(rounded_perc)
    pyplot.title('Masked Forged Region')
    pyplot.imshow(np.round(np.expand_dims(mask, axis=-1) * img[::1, ::1]).astype('uint8'), cmap='jet')
    pyplot.savefig('static/uploads/saved.jpg')
    return rounded_perc


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route("/upload_url", methods=['GET', 'POST'])
def upload_url():
    if request.method == 'POST':
        image_url = request.form['search']
        urllib.request.urlretrieve(image_url, 'static/uploads/test.jpg')
        return render_template("1.html")


@app.route("/upload_file", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file.save('./static/uploads/test.jpg')
        return render_template('1.html')


@app.route("/upload", methods=['GET', 'POST'])
def upload_for_processing():
    with graph.as_default():
        if request.method == 'POST':
            perc = get_image()
            print("Percentage")
            print(perc)
            # return render_template("result.html")
            return render_template('result.html', perc=perc)


if __name__ == '__main__':
    app.run(debug=True)

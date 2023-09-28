from flask import Flask, render_template, request, flash
from keras.models import load_model

import numpy as np
import os
from PIL import Image
from datetime import datetime
from keras.preprocessing import image

app = Flask(__name__)
app.secret_key="qwerty098765421"

# load model for prediction
modeldensenet201 = load_model("resnanda-rempah-densenet-98-08.h5")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("classifications.html")

@app.route('/submit', methods=['POST'])
def predict():
    files = request.files.getlist('file')
    filename = "temp_image.png"
    # errors = {}
    success = False
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            flash("Anda Belum Mengunggah File atau Ekstensi File Salah, \
                  Silahkan Ulangi Unggah File dan Pastikan Ekstensi File Sudah Sesuai Panduan di Atas!")
            return render_template("classifications.html")
    
    # if not success:
    #     resp = jsonify(errors)
    #     resp.status_code = 400
    #     return resp
    img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # convert image to RGB
    img = Image.open(img_url).convert('RGB')
    now = datetime.now()
    predict_image_path = 'static/uploads/' + now.strftime("%d%m%y-%H%M%S") + ".png"
    image_predict = predict_image_path
    img.convert('RGB').save(image_predict, format="png")
    img.close()

    # prepare image for prediction
    img = image.load_img(predict_image_path, target_size=(256, 256))
    x = image.img_to_array(img)/255.0
    x = x.reshape(1, 256, 256,3)
    images = np.array(x)

    # predict
    prediction_array_densenet201 = modeldensenet201.predict(images)

    # prepare api response
    class_names = ['Jahe', 'Kunyit', 'Lengkuas']
	
    return render_template("classifications.html", img_path = img_url,
                        predictiondensenet201 = class_names[np.argmax(prediction_array_densenet201)],
                        confidencedensenet201 = '{:2.0f}%'.format(100 * np.max(prediction_array_densenet201))
                        )

if __name__ =='__main__':
	#app.debug = True
    app.run(debug = True)
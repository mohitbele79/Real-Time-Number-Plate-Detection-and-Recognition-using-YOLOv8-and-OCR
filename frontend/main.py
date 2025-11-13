from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from detector import detect_number_plate


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image = request.files.get('image')
        video = request.files.get('video')

        uploaded_path = None
        processed_path = None
        file_type = None
        number = None

        if image and image.filename != '':
            filename = secure_filename(image.filename)
            uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(uploaded_path)

            processed_full_path, number = detect_number_plate(uploaded_path)
            processed_path = processed_full_path.replace("\\", "/")
            file_type = 'image'

        elif video and video.filename != '':
            filename = secure_filename(video.filename)
            uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video.save(uploaded_path)

            processed_full_path, number = detect_number_plate(uploaded_path)
            processed_path = processed_full_path.replace("\\", "/")
            file_type = 'video'

        return render_template('predict.html',
                               uploaded_path=f"/static/uploads/{filename}",
                               processed_path=processed_path,
                               file_type=file_type,
                               number=number)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)

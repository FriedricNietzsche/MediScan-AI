from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
import csv
from datetime import datetime
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
# from pushbullet import PushBullet
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input


# Loading Models (wrapped so app import won't crash if TF / protobuf versions mismatch)
covid_model = None
braintumor_model = None
alzheimer_model = None
diabetes_model = None
heart_model = None
pneumonia_model = None
breastcancer_model = None

def try_load_models():
    global covid_model, braintumor_model, alzheimer_model, diabetes_model, heart_model, pneumonia_model, breastcancer_model
    try:
        covid_model = load_model('models/covid.h5')
    except Exception as e:
        print('Warning: could not load covid model:', e)
        covid_model = None
    try:
        braintumor_model = load_model('models/braintumor.h5')
    except Exception as e:
        print('Warning: could not load braintumor model:', e)
        braintumor_model = None
    try:
        alzheimer_model = load_model('models/alzheimer_model.h5')
    except Exception as e:
        print('Warning: could not load alzheimer model:', e)
        alzheimer_model = None
    try:
        diabetes_model = pickle.load(open('models/diabetes.sav', 'rb'))
    except Exception as e:
        print('Warning: could not load diabetes model:', e)
        diabetes_model = None
    try:
        heart_model = pickle.load(open('models/heart_disease.pickle.dat', "rb"))
    except Exception as e:
        print('Warning: could not load heart model:', e)
        heart_model = None
    try:
        pneumonia_model = load_model('models/pneumonia_model.h5')
    except Exception as e:
        print('Warning: could not load pneumonia model:', e)
        pneumonia_model = None
    try:
        breastcancer_model = joblib.load('models/cancer_model.pkl')
    except Exception as e:
        print('Warning: could not load breast cancer model:', e)
        breastcancer_model = None

# Attempt to load models now; failures won't stop Flask from importing.
try_load_models()

# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


# Detection history file
HISTORY_FILE = 'detection_history.csv'

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def interpret_binary_prediction(pred):
    """Normalize various Keras prediction outputs to (class_idx, prob_for_class1).
    - If pred is a scalar or shape (1,): treat as probability for class 1.
    - If pred has length 2 (softmax), take argmax as class index and prob_for_class1 = prob at index 1.
    Returns (0 or 1, prob)
    """
    arr = np.array(pred)
    arr = arr.reshape(-1)
    if arr.size == 1:
        prob1 = float(arr[0])
        cls = 1 if prob1 >= 0.5 else 0
        return cls, prob1
    elif arr.size == 2:
        # assume softmax [prob_class0, prob_class1]
        prob0, prob1 = float(arr[0]), float(arr[1])
        cls = int(prob1 >= prob0)
        return cls, prob1
    else:
        # unexpected shape: fall back to argmax
        idx = int(np.argmax(arr))
        prob1 = float(arr[1]) if arr.size > 1 else float(arr[idx])
        return idx, prob1

def log_detection(user, model, result):
    # user: dict with keys firstname, lastname, email, phone, gender, age
    # model: string
    # result: string/int
    # Convert binary results to human-readable
    readable_result = result
    if model in ['COVID-19', 'Brain Tumor', 'Diabetes', 'Breast Cancer', 'Pneumonia', 'Heart Disease']:
        # Accept int, float, numpy types, string
        try:
            val = int(result)
        except Exception:
            try:
                val = int(float(result))
            except Exception:
                val = str(result)
        if val == 1:
            readable_result = 'Positive'
        elif val == 0:
            readable_result = 'Negative'
        else:
            readable_result = str(result)
    if model == 'Alzheimer':
        alz_map = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        try:
            readable_result = alz_map[int(result)]
        except:
            readable_result = str(result)
    # Ensure file exists before appending
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            pass
    with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user.get('firstname', ''),
            user.get('lastname', ''),
            user.get('email', ''),
            user.get('phone', ''),
            user.get('gender', ''),
            user.get('age', ''),
            model,
            readable_result
        ])

# Route to clear detection history
@app.route('/clear_history', methods=['POST'])
def clear_history():
    # Always flush the file after clearing
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        pass
    flash('Detection history cleared.')
    return redirect(url_for('history'))

############################################# BRAIN TUMOR FUNCTIONS ################################################

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

########################### Routing Functions ########################################

@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/covid')
def covid():
    return render_template('covid.html')


@app.route('/breastcancer')
def breast_cancer():
    return render_template('breastcancer.html')


@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')


@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')


@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')


########################### Result Functions ########################################


@app.route('/resultc', methods=['POST'])
def resultc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            img = img/255.0
            if covid_model is None:
                flash('COVID model is not available on the server right now. See server logs for details.')
                return redirect('/covid')
            raw = covid_model.predict(img)
            cls, prob = interpret_binary_prediction(raw)
            # For COVID, invert the class because model outputs high prob for NORMAL (negative)
            pred = 1 - cls  # If cls=0 (NORMAL), pred=1 (Positive); if cls=1 (COVID), pred=0 (Negative)
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            log_detection({
                'firstname': firstname,
                'lastname': lastname,
                'email': email,
                'phone': phone,
                'gender': gender,
                'age': age
            }, 'COVID-19', pred)
            return render_template('resultc.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resultbt', methods=['POST'])
def resultbt():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = crop_imgs([img])
            img = img.reshape(img.shape[1:])
            img = preprocess_imgs([img], (224, 224))
            if braintumor_model is None:
                flash('Brain Tumor model is not available on the server right now. See server logs for details.')
                return redirect('/braintumor')
            pred = braintumor_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Brain Tumor test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
            log_detection({
                'firstname': firstname,
                'lastname': lastname,
                'email': email,
                'phone': phone,
                'gender': gender,
                'age': age
            }, 'Brain Tumor', pred)
            return render_template('resultbt.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resultd', methods=['POST'])
def resultd():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetespedigree = request.form['diabetespedigree']
        age = request.form['age']
        skinthickness = request.form['skin']
        if diabetes_model is None:
            flash('Diabetes model is not available on the server right now. See server logs for details.')
            return redirect('/diabetes')
        pred = diabetes_model.predict(
            [[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]])
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        log_detection({
            'firstname': firstname,
            'lastname': lastname,
            'email': email,
            'phone': phone,
            'gender': gender,
            'age': age
        }, 'Diabetes', pred)
        return render_template('resultd.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


@app.route('/resultbc', methods=['POST'])
def resultbc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        cpm = request.form['concave_points_mean']
        am = request.form['area_mean']
        rm = request.form['radius_mean']
        pm = request.form['perimeter_mean']
        cm = request.form['concavity_mean']
        if breastcancer_model is None:
            flash('Breast cancer model is not available on the server right now. See server logs for details.')
            return redirect('/breastcancer')
        pred = breastcancer_model.predict(
            np.array([cpm, am, rm, pm, cm]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Breast Cancer test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        log_detection({
            'firstname': firstname,
            'lastname': lastname,
            'email': email,
            'phone': phone,
            'gender': gender,
            'age': age
        }, 'Breast Cancer', pred)
        return render_template('resultbc.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


@app.route('/resulta', methods=['GET', 'POST'])
def resulta():
    if request.method == 'POST':
        print(request.url)
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (176, 176))
            img = img.reshape(1, 176, 176, 3)
            img = img/255.0
            if alzheimer_model is None:
                flash('Alzheimer model is not available on the server right now. See server logs for details.')
                return redirect('/alzheimer')
            pred = alzheimer_model.predict(img)
            pred = pred[0].argmax()
            # Remap due to class order mismatch in training
            remap = {0: 2, 1: 3, 2: 0, 3: 1}  # Model's output to correct alz_map index
            pred = remap.get(pred, pred)
            print("Predicted class index:", pred)
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Alzheimer test results are ready.\nRESULT: {}'.format(firstname,['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'][pred]))
            log_detection({
                'firstname': firstname,
                'lastname': lastname,
                'email': email,
                'phone': phone,
                'gender': gender,
                'age': age
            }, 'Alzheimer', pred)
            return render_template('resulta.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')


@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (150, 150))
            img = img.reshape(1, 150, 150, 3)
            img = img/255.0
            if pneumonia_model is None:
                flash('Pneumonia model is not available on the server right now. See server logs for details.')
                return redirect('/pneumonia')
            pred = pneumonia_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            log_detection({
                'firstname': firstname,
                'lastname': lastname,
                'email': email,
                'phone': phone,
                'gender': gender,
                'age': age
            }, 'Pneumonia', pred)
            return render_template('resultp.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        if heart_model is None:
            flash('Heart disease model is not available on the server right now. See server logs for details.')
            return redirect('/heartdisease')
        pred = heart_model.predict(
            np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        log_detection({
            'firstname': firstname,
            'lastname': lastname,
            'email': email,
            'phone': phone,
            'gender': gender,
            'age': age
        }, 'Heart Disease', pred)
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)
# New route for history page
@app.route('/history')
def history():
    history_data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                history_data.append(row)
    return render_template('history.html', history=history_data)

# New route for resources page
@app.route('/resources')
def resources():
    return render_template('resources.html')


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)

import sys, cv2, numpy as np
import app
print('covid_model is', type(app.covid_model))
if app.covid_model is None:
    print('COVID model is not loaded')
    sys.exit(0)
img_path = 'preview/aug_img_0_1128.jpg'
img = cv2.imread(img_path)
if img is None:
    print('Image', img_path, 'not found, trying static/uploads sample')
    import os
    up = 'static/uploads'
    files = [f for f in os.listdir(up) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not files:
        print('No upload images found to test.')
        sys.exit(0)
    img = cv2.imread(os.path.join(up, files[0]))
    print('Using', files[0])
img = cv2.resize(img, (224,224))
img = img.reshape(1,224,224,3)/255.0
pred = app.covid_model.predict(img)
print('raw pred:', pred, 'type:', type(pred))
try:
    val = float(pred)
except Exception:
    val = float(np.array(pred).reshape(-1)[0])
print('scalar value:', val)
print('interpreted label:', 'Positive' if val>=0.5 else 'Negative')

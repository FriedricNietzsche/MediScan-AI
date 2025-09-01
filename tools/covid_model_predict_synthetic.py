import importlib.util
import sys
from pathlib import Path
import numpy as np

proj = Path(__file__).resolve().parents[1]
app_path = proj / 'app.py'
if not app_path.exists():
    print('app.py not found at', app_path)
    sys.exit(1)

spec = importlib.util.spec_from_file_location('app_from_path', str(app_path))
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

print('covid_model object:', type(app.covid_model))
if app.covid_model is None:
    print('COVID model is not loaded. Check server logs or reinstall correct TF version per README.')
    sys.exit(0)

# create synthetic input matching expected shape 224x224x3
x = np.random.rand(1,224,224,3).astype('float32')
raw = app.covid_model.predict(x)
print('raw prediction:', raw)
# use helper if available
if hasattr(app, 'interpret_binary_prediction'):
    cls, prob = app.interpret_binary_prediction(raw)
    print('interpreted class:', cls, 'prob:', prob)
else:
    print('No helper interpret_binary_prediction available in app module')

print('done')

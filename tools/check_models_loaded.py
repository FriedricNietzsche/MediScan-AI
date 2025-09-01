import importlib
app = importlib.import_module('app')
print('covid_model loaded:', type(app.covid_model))
print('braintumor_model loaded:', type(app.braintumor_model))
print('alzheimer_model loaded:', type(app.alzheimer_model))
print('breastcancer_model loaded:', type(app.breastcancer_model))
print('diabetes_model loaded:', type(app.diabetes_model))
print('heart_model loaded:', type(app.heart_model))
print('pneumonia_model loaded:', type(app.pneumonia_model))

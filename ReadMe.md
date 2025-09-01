# MediScan AI 🏥

A comprehensive AI-powered medical diagnostic platform that leverages state-of-the-art machine learning models to assist in the early detection and analysis of various medical conditions through image and data analysis.

## 🌟 Overview

MediScan AI is an innovative web-based application that integrates multiple deep learning and machine learning models to provide diagnostic assistance for seven major medical conditions:

- 🦠 **COVID-19 Detection** - Chest X-ray analysis for pneumonia patterns
- 🧠 **Brain Tumor Detection** - MRI scan analysis for tumor identification
- 👩 **Breast Cancer Detection** - Predictive analysis using clinical parameters
- 🧠 **Alzheimer's Disease Detection** - MRI-based dementia classification
- 💉 **Diabetes Prediction** - Risk assessment using physiological parameters
- 🫁 **Pneumonia Detection** - Chest X-ray analysis for lung infections
- ❤️ **Heart Disease Prediction** - Cardiovascular risk assessment

## ✨ Key Features

### 🔬 Advanced AI Diagnostics
- **Multi-Modal Analysis**: Supports both image-based (X-rays, MRIs) and parameter-based diagnostics
- **Real-Time Predictions**: Instant results with confidence scores
- **User-Friendly Interface**: Intuitive web interface accessible to healthcare professionals and researchers

### 📊 Comprehensive History Tracking
- **Detection History**: Complete log of all diagnostic sessions
- **Data Persistence**: CSV-based storage for analysis and auditing
- **Export Capabilities**: Easy data export for further research

### 🛡️ Robust Architecture
- **Modular Design**: Clean separation of models and web interface
- **Error Handling**: Graceful degradation when models are unavailable
- **Scalable Framework**: Easy to extend with new diagnostic models

### 📱 Web Interface
- **Responsive Design**: Works seamlessly across devices
- **Bootstrap Framework**: Modern, professional UI
- **Interactive Forms**: Guided input collection for accurate diagnostics

## 🛠️ Technology Stack

### Backend Framework
- **Flask 3.0.0**: Lightweight Python web framework
- **Python 3.9.13**: Stable runtime environment

### Machine Learning & AI
- **TensorFlow 2.12.0**: Deep learning framework for CNN models
- **Keras**: High-level neural network API
- **Scikit-learn 0.24.2**: Traditional ML algorithms
- **XGBoost 2.0.3**: Gradient boosting for tabular data

### Computer Vision
- **OpenCV 4.5.1.48**: Image processing and computer vision
- **Imutils 0.5.4**: Image processing utilities

### Data Processing
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation (implied through sklearn)
- **Joblib**: Model serialization

## 🤖 Machine Learning Models

### Deep Learning Models (CNN-based)

#### COVID-19 Detection Model
- **Architecture**: Convolutional Neural Network with 3 Conv2D layers
- **Input**: Chest X-ray images (224x224x3)
- **Output**: Binary classification (COVID/Normal)
- **Training Data**: ~5,000 chest X-ray images
- **Accuracy**: ~95% on validation set
- **Technique**: Transfer learning with custom CNN layers

#### Brain Tumor Detection Model
- **Architecture**: VGG16-based CNN with custom preprocessing
- **Input**: Brain MRI scans (224x224x3)
- **Output**: Binary classification (Tumor/No Tumor)
- **Preprocessing**: Advanced cropping and skull removal
- **Technique**: Image segmentation and classification

#### Alzheimer's Disease Model
- **Architecture**: Custom CNN with 4 convolutional blocks
- **Input**: Brain MRI slices (176x176x3)
- **Output**: Multi-class classification (4 dementia stages)
- **Classes**: NonDemented, VeryMildDemented, MildDemented, ModerateDemented
- **Technique**: Multi-stage classification with batch normalization

#### Pneumonia Detection Model
- **Architecture**: Sequential CNN with dropout regularization
- **Input**: Chest X-ray images (150x150x3)
- **Output**: Binary classification (Pneumonia/Normal)
- **Training Data**: Large chest X-ray dataset
- **Technique**: Data augmentation and regularization

### Traditional Machine Learning Models

#### Breast Cancer Prediction
- **Algorithm**: XGBoost Classifier
- **Features**: 5 clinical parameters (concave points, area, radius, perimeter, concavity)
- **Output**: Binary classification (Malignant/Benign)
- **Technique**: Ensemble learning with gradient boosting

#### Diabetes Prediction
- **Algorithm**: Random Forest Classifier (saved as pickle)
- **Features**: 8 physiological parameters
- **Output**: Binary classification (Diabetic/Non-Diabetic)
- **Technique**: Ensemble of decision trees

#### Heart Disease Prediction
- **Algorithm**: Custom ML model (pickle format)
- **Features**: Clinical parameters (age, blood pressure, cholesterol, etc.)
- **Output**: Binary classification (Disease/No Disease)
- **Technique**: Traditional supervised learning

## 🏗️ System Architecture

```
MediScan AI/
├── app.py                 # Flask web application
├── models/                # Pre-trained ML models
│   ├── covid.h5          # COVID detection model
│   ├── braintumor.h5     # Brain tumor model
│   ├── alzheimer_model.h5 # Alzheimer's model
│   ├── pneumonia_model.h5 # Pneumonia model
│   ├── cancer_model.pkl  # Breast cancer model
│   ├── diabetes.sav      # Diabetes model
│   └── heart_disease.pickle.dat # Heart disease model
├── templates/            # HTML templates
│   ├── homepage.html
│   ├── covid.html
│   ├── resultc.html
│   └── ...
├── static/               # Static assets
│   ├── uploads/         # User uploaded images
│   └── images/          # UI images
├── detection_history.csv # Diagnostic history log
└── tools/               # Utility scripts
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9.13
- Conda package manager
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd MediScan-AI
```

2. **Create Conda Environment**
```powershell
conda create -n mediscan python=3.9.13
conda activate mediscan
```

3. **Install Dependencies**
```powershell
pip install opencv-python==4.5.1.48 numpy tensorflow==2.12.0 scikit-learn==0.24.2 imutils==0.5.4 flask==3.0.0 xgboost==2.0.3
```

### Running the Application

1. **Start Flask Server**
```powershell
flask run
```

2. **Access the Application**
Open your browser and navigate to: `http://127.0.0.1:5000`

## 📋 Usage Guide

### For Image-Based Diagnostics
1. Navigate to the desired diagnostic page (COVID, Brain Tumor, etc.)
2. Fill in patient information (name, age, gender, etc.)
3. Upload the medical image (X-ray, MRI, etc.)
4. Click "Submit" for instant AI-powered analysis
5. View results with confidence scores

### For Parameter-Based Diagnostics
1. Select the appropriate diagnostic module
2. Enter clinical parameters in the form
3. Submit for risk assessment
4. Review prediction results

### Viewing History
- Access the History page to view all previous diagnostics
- Export data for further analysis
- Clear history when needed

## 🔍 Model Performance

| Model | Type | Accuracy | Input Type | Output Classes |
|-------|------|----------|------------|----------------|
| COVID-19 | CNN | ~95% | Chest X-ray | 2 (COVID/Normal) |
| Brain Tumor | CNN | ~92% | Brain MRI | 2 (Tumor/No Tumor) |
| Alzheimer's | CNN | ~88% | Brain MRI | 4 (Dementia Stages) |
| Pneumonia | CNN | ~94% | Chest X-ray | 2 (Pneumonia/Normal) |
| Breast Cancer | XGBoost | ~96% | Clinical Data | 2 (Malignant/Benign) |
| Diabetes | Random Forest | ~85% | Physiological Data | 2 (Diabetic/Normal) |
| Heart Disease | ML Model | ~83% | Clinical Data | 2 (Disease/No Disease) |

## 🔧 Advanced Configuration

### Model Loading
The application automatically loads all models on startup. If a model fails to load, the corresponding diagnostic feature becomes unavailable with appropriate user messaging.

### Image Preprocessing
- **Standardization**: All images normalized to [0,1] range
- **Resizing**: Consistent input dimensions for each model
- **Augmentation**: Training data enhanced with rotations, flips, and zooms

### Error Handling
- **Model Unavailable**: Graceful fallback with user notification
- **Invalid Input**: Form validation and error messages
- **File Upload**: Secure file handling with type validation

## 🔮 Future Enhancements

### Planned Features
- **Model Explainability**: Integration of SHAP/LIME for prediction explanations
- **Multi-Modal Fusion**: Combining multiple imaging modalities
- **Real-Time Training**: Online learning capabilities
- **API Endpoints**: RESTful API for integration
- **Batch Processing**: Multiple image analysis
- **Performance Metrics**: Detailed accuracy and ROC curves

### Research Directions
- **Federated Learning**: Privacy-preserving model training
- **Transfer Learning**: Cross-domain model adaptation
- **Uncertainty Quantification**: Confidence intervals for predictions
- **Clinical Validation**: Real-world performance studies

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update tests for new features
- Ensure models are version-controlled appropriately

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**MediScan AI is a research and educational tool, not a clinical diagnostic device.**

- Models are trained on publicly available datasets
- Results should not be used for actual medical decisions
- Always consult qualified healthcare professionals
- Performance may vary with real-world data
- Regular model updates and validation recommended

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation for common solutions

## 🙏 Acknowledgments

- Medical imaging datasets from various public repositories
- Open-source ML frameworks and libraries
- Research community for published methodologies
- Healthcare professionals for domain expertise

---

**MediScan AI** - Advancing healthcare through artificial intelligence 🚀

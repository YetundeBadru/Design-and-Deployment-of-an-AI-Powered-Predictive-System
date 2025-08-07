# Design and Deployment of a Predictive System : 
### Heart Disease Predictor
An AI-powered predictive system that helps assess the likelihood of heart disease using clinical data and a user-friendly web interface.

---

*Date:* 13/06/2025

---

![Heart](https://img.shields.io/badge/ML-Predictive_Model-green?style=for-the-badge)  
![Flask](https://img.shields.io/badge/Flask-Deployed-blue?style=for-the-badge)  
![Deployment](https://img.shields.io/badge/Render-Deployment-success?style=for-the-badge)

---

## Project Summary 

The project involves developing and deploying a machine learning or AI-based model to predict heart disease based on clinical data. The solution includes data preprocessing, model development, and deployment using Flask so it can be accessed via a simple and user-friendly web interface.

AI models can be developed by collecting relevant real-world data, preprocessing it to ensure quality, and training a machine learning algorithm to make predictions or classifications based on the data. Once trained, the model can be evaluated using standard performance metrics. To deploy the model in a real-world application, a web framework like Flask can be used to create a user-friendly interface where inputs are collected, passed to the model, and predictions are returned in real time. This approach enhances decision-making and task automation by enabling quick, data-driven responses.

---

## Project Objectives:

1. Explore and analyze the Heart Disease UCI dataset.
2. Build and evaluate a supervised ML model.
3. Develop a web interface using Flask.
4. Deploy the model for real-time predictions.

---

## Dataset Detailed Summary

- **Source**: [Heart Disease UCI – Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
- **Target**: num (Presence of heart disease (`0 = No`, `1 = Yes`))
- **Features**: Age, sex, chest pain type, resting blood pressure, cholesterol, etc.

---

## Tools & Technologies

- **Python**: Data processing and model building
- **Pandas, NumPy, Seaborn, Matplotlib**: EDA and visualization
- **Scikit-learn**: ML algorithms, metrics, preprocessing
- **Random Forest Classifier**: Main prediction model
- **Flask**: Web framework
- **Render**: Cloud deployment platform

---

## Project Workflow

### Phase 1: Dataset & Problem Definition

*About Dataset*

The World Health Organization has estimated 12 million deaths occur worldwide, every year due to heart diseases. Half the deaths in the United States and other developed countries are due to cardiovascular diseases. The early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high-risk patients and in turn reduce the complications. This research intends to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using Random Forest Model.

*Meta-Data*

This is a multivariate type of dataset which means providing or involving a variety of separate mathematical or statistical variables, multivariate numerical data analysis. It is composed of 14 attributes which are age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, old-peak - ST depression induced by exercise relative to rest, the slope of the peak exercise ST segment, number of major vessels and Thalassemia. This database includes 76 attributes, but all published studies relate to the use of a subset of 14 of them. The Cleveland database is the only one used by ML researchers to date. One of the major tasks on this dataset is to predict based on the given attributes of a patient whether that particular person has heart disease or not and another is the experimental task to diagnose and find out various insights from this dataset which could help in understanding the problem more.

*Column Descriptions:*

- id (Unique id for each patient)
- age (Age of the patient in years)
- origin (place of study)
- sex (Male/Female)
- cp chest pain type ([typical angina, atypical angina, non-anginal, asymptomatic])
- trestbps resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
- chol (serum cholesterol in mg/dl)
- fbs (if fasting blood sugar > 120 mg/dl)
- restecg (resting electrocardiographic results) Values: [normal, stt abnormality, lv hypertrophy]
- thalach: maximum heart rate achieved
- exang: exercise-induced angina (True/ False)
- oldpeak: ST depression induced by exercise relative to rest
- slope: the slope of the peak exercise ST segment
- ca: number of major vessels (0-3) colored by fluoroscopy
- thal: [normal; fixed defect; reversible defect]
- num: the predicted attribute

---

### Phase 2: Data Exploration and Preprocessing
- Loaded and cleaned the dataset
- Explored data types, value ranges, and missing values.
- Feature scaling with `StandardScaler`

### Phase 3: Model Development
- Target separation (`X`, `y`)
- Train-test split: 80/20
- Trained a Model: `RandomForestClassifier`
- Evaluated using Accuracy, Precision, Recall, F1-score
- Saved using: `.pkl` files for model and scaler using `joblib`

### Phase 4: Web App Development
- Built with Flask
- `index.html`: Input form
- `result.html`: Prediction output
- User inputs are preprocessed and passed to the model for prediction
- Displayed prediction on a result page

### Phase 5: Deployment on Render
- Deployed the Flask-based heart disease prediction app using Render’s Blueprint instance
- Created and configured a render.yaml file for infrastructure-as-code deployment
- Defined buildCommand and startCommand for clean automated builds
- Added requirements.txt to install dependencies and ensured .pkl files (model + scaler) were available
- HTML templates (index.html, result.html) served via Flask’s templates directory
- Connected the GitHub repo for automatic deployment on new commits
- Successfully deployed to a public Render URL with live prediction functionality

---
## How to Run Locally
<pre>```bash
# Clone the repo
git clone https://github.com/yourusername/heart-disease-predictor.git

# Navigate into the folder
cd heart-disease-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open http://127.0.0.1:5000 in your browser
</pre>
---


## 🌐 Live App
The live app is hosted on Render:
🔗 [Access the Web App on Render](https://heart-disease-app-sxju.onrender.com)

---

## 📷 Demo Screenshots

**Homepage**

<img width="958" height="682" alt="image" src="https://github.com/user-attachments/assets/362ee216-35b2-4ee7-9f92-1f8bb1ace36b" />

---

**Prediction Result**

<img width="1147" height="658" alt="image" src="https://github.com/user-attachments/assets/9d51af4e-50e2-4fcc-a948-dc8b41635f1d" />

---

## Project Structure

heart-disease-prediction-app/

│

├── app.py                      

├── heart_disease_model.pkl    

├── scaler.pkl                

├── requirements.txt          

├── render.yaml               

│

├── templates/               

│   ├── index.html          

│   └── result.html          

│

└── README.md                

---

## Future Improvements
- Expand model testing on larger datasets
- Integrate multiple ML models for comparison
- Add user authentication and history tracking
- Use Docker for more portable deployment

---

## Documentation
Full project documentation with background, objectives, methods, results, and discussion is available within the notebook and linked below:

🔗 [Project Documentation](https://docs.google.com/document/d/1gXhbhJin4VRKeuCtvIuj386eWWgKeSEON3u1QOSRnac/edit?usp=sharing)

---

## Acknowledgments
- Dataset from UCI via Kaggle
- Tools from Python’s data science ecosystem
- Deployment powered by Render

---

## Contributing
Feel free to fork this repository, contribute, and suggest improvements.

---

## Author
**Yetunde Badru**
Data Scientist | AI/ML | AWS Cloud Practitioner

**LinkedIn**: www.linkedin.com/in/yetundebarakbadru

**Email:** yetundebarakbadru@gmail.com



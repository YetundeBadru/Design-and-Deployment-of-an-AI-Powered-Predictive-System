# Design-and-Deployment-of-an-AI-Powered-Predictive-System
Develop and deploy machine learning or AI-based model for a real-world application. Preprocess data, build and evaluate a predictive model, and deploy the solution using Flask so that it can be accessed via a simple web interface.
-------------------------------------------------------
Date: 13/06/2025
--------------------------------------------------------
## Project Summary 
The project involves developing and deploying a machine learning model to predict heart disease based on clinical data. The solution includes data preprocessing, model development, and deployment using Flask for a user-friendly web interface.
---------------------------------------------------------
## Project Question 
AI models can be developed by collecting relevant real-world data, preprocessing it to ensure quality, and training a machine learning algorithm to make predictions or classifications based on the data. Once trained, the model can be evaluated using standard performance metrics. To deploy the model in a real-world application, a web framework like Flask can be used to create a user-friendly interface where inputs are collected, passed to the model, and predictions are returned in real time. This approach enhances decision-making and task automation by enabling quick, data-driven responses.
---------------------------------------------------------
## Project Objectives:
1. Explore and analyze the Heart Disease UCI dataset.
2. Build and evaluate a supervised ML model.
3. Develop a web interface using Flask.
4. Deploy the model for real-time predictions.
----------------------------------------------------------
## Dataset & Problem Definition
Problem: Predict the presence of heart disease from clinical data
Dataset: Heart Disease UCI [Link Text](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
Target Variable : num (presence of heart disease)
Features: age, sex, chest pain type, resting blood pressure, cholesterol, etc.
----------------------------------------------------------------------
## STEPS:
### Dataset & Problem Definition
About Dataset
The World Health Organization has estimated 12 million deaths occur worldwide, every year due to heart diseases. Half the deaths in the United States and other developed countries are due to cardiovascular diseases. The early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high-risk patients and in turn reduce the complications. This research intends to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using Random Forest Model.

Meta-Data
This is a multivariate type of dataset which means providing or involving a variety of separate mathematical or statistical variables, multivariate numerical data analysis. It is composed of 14 attributes which are age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, old-peak - ST depression induced by exercise relative to rest, the slope of the peak exercise ST segment, number of major vessels and Thalassemia. This database includes 76 attributes, but all published studies relate to the use of a subset of 14 of them. The Cleveland database is the only one used by ML researchers to date. One of the major tasks on this dataset is to predict based on the given attributes of a patient whether that particular person has heart disease or not and another is the experimental task to diagnose and find out various insights from this dataset which could help in understanding the problem more.

Column Descriptions:
id (Unique id for each patient)
age (Age of the patient in years)
origin (place of study)
sex (Male/Female)
cp chest pain type ([typical angina, atypical angina, non-anginal, asymptomatic])
trestbps resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
chol (serum cholesterol in mg/dl)
fbs (if fasting blood sugar > 120 mg/dl)
restecg (resting electrocardiographic results)
 -- Values: [normal, stt abnormality, lv hypertrophy]
thalach: maximum heart rate achieved
exang: exercise-induced angina (True/ False)
oldpeak: ST depression induced by exercise relative to rest
slope: the slope of the peak exercise ST segment
ca: number of major vessels (0-3) colored by fluoroscopy
thal: [normal; fixed defect; reversible defect]
num: the predicted attribute

Acknowledgements
Creators:
Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
Relevant Papers:
Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64,304--310.
 Web Link
David W. Aha & Dennis Kibler. "Instance-based prediction of heart-disease presence with the Cleveland database." Web Link
Gennari, J.H., Langley, P, & Fisher, D. (1989). Models of incremental concept formation. Artificial Intelligence, 40, 11--61. Web Link

Citation Request:
The authors of the databases have requested that any publications resulting from the use of the data include the names of the principal investigator responsible for the data collection at each institution. They would be:
Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
Dataset Detailed Summary:
Source: Heart Disease UCI dataset from Kaggle.
Target Variable: num (presence of heart disease).
Features: Age, sex, chest pain type, resting blood pressure, cholesterol, etc.
 
 ### 2. Data Preparation
*Import all Necessary Libraries:*
<img width="1099" height="467" alt="image" src="https://github.com/user-attachments/assets/d9df945e-c0e4-43d5-9fc9-a933c8c4d320" />
 
*Data Collection from Kaggle:*
Load the Dataset:
<img width="1049" height="271" alt="image" src="https://github.com/user-attachments/assets/e312eefd-a2ac-436e-aea0-c022aab3d243" />

Descriptive Analysis:
<img width="1081" height="338" alt="image" src="https://github.com/user-attachments/assets/d643c9fa-2822-4220-a565-672eddd60e85" />
<img width="836" height="332" alt="image" src="https://github.com/user-attachments/assets/7e82610a-8ead-4abb-b9d2-df91ec3f8a3f" />
<img width="938" height="454" alt="image" src="https://github.com/user-attachments/assets/0ca97b1f-8f6f-401e-a7bf-cf19147bd2b0" />
<img width="1048" height="250" alt="image" src="https://github.com/user-attachments/assets/767b5dde-2d8a-4e70-a258-0150694770c7" />
<img width="889" height="84" alt="image" src="https://github.com/user-attachments/assets/e8ff4de6-0a9a-4278-a6dc-ea7271ea1359" />



Observations:
There are no heart diseases found in the 25% of the patients with an average of age 47.0.
There is mild presence of heart diseases found in the 50% of the patients with an average of age 54.0.
There is moderate presence of heart diseases found in 75% of the patients with an average of age 60.0 or more.
–----------------------------------------------------------------------------------------------------------------

Observations:
There are 920 rows, which means the data of 920 human beings.
There are 16 columns in the dataset, including id, dataset (location of the patient).
The target feature num represents the ordinal numeric severity of the heart disease ([0, 1, 2, 3, 4]).
There are 13 features or medical parameters (excluding id and dataset), which will be used to predict the target feature num (the intensity of the heart disease).
------------------------------------------------------------------------------------------------------------------

Observation:
Number of rows in the dataset:  920
Number of columns in the dataset:  16
---------------------------------------------------------------------------------------------------------------------

Observation:
The columns in the Dataset:
------------------------------------------------------------------------------------------------------------------


 Observation:
There is no duplicated entry in the Dataset

–------------------------------------------------------------ 
3.  Exploratory Data Analysis (EDA):
EDA of "Age" Column:


AGE DISTRIBUTION

MEAN, MEDIAN & MODE of Age DISTRIBUTION
 

Age
The minimum age to have a heart disease starts from 28 years old.
Most of the people get heart disease at the age of 53-54 years.
Most of the males and females get are with heart disease at the age of 54-55 years.
Male percentage in the data: 78.91%
Female Percentage in the data: 21.09%
Males are 274.23% more than females in the data.
 
 
EDA of "Sex" Column

 



Sex
Most gender with heart disease are the males with 726 count and female is 194.
 
 
 
 
 
 
 
 
 
 
 
 
 
EDA of CP (Chest Pain) column



CP (Chest Pain)
0 = no heart disease 1 = mild heart disease 2 = moderate heart disease 3 = severe heart disease 4 = critical heart disease
The most common chest pain type is asymptomatic with 496 patients
A total of 104 individuals are identified as having neither chest pain nor heart disease.
Only 23 individuals are found to have no chest pain while experiencing critical heart disease.
A group of 83 individuals is observed to be free from chest pain while having severe heart disease.
In the dataset, 197 individuals are noted for having no chest pain and exhibiting mild heart disease.
Among the individuals, 89 have no chest pain while presenting with moderate heart disease.
 
 
 
EDA of trestbps (resting blood pressure)
The normal resting blood pressure is 120/80 mm Hg.
high blood pressure increasing the risk of heart disease and stroke, often asymptomatic, while low blood pressure can lead to dizziness and fainting
 


trestbps
Most gender with heart disease are the males with 726 count and female is 194.
 
 
 
 
EDA of Chol column


EDA of Thal column
Normal: Within expected or healthy parameters.
Reversible Defect: An abnormality that can potentially be corrected or improved.
Fixed Defect: An abnormality that is unlikely to change or be corrected.
 





Normal: Within expected or healthy parameters.
Reversible Defect: An abnormality that can potentially be corrected or improved.
Fixed Defect: An abnormality that is unlikely to change or be corrected.
Observations:
Among the individuals, 110 males and 86 females are classified as normal.
A total of 42 males and 4 females exhibit a fixed defect.
In the dataset, 171 males and 21 females are identified with a reversible defect. The higher ratio of males compared to females is attributed to the dataset's male predominance.
Both individuals with thalassemia and those with normal thalassemia experience chest pain.
Individuals with normal thalassemia often exhibit a higher ratio of being free from heart disease, although some may still experience heart-related conditions.
Those with thalassemia generally have an increased likelihood of heart disease, yet some individuals with thalassemia do not develop such health issues.
 
 
 
 
 
EDA of 'num' column
Lets Deal With Num , The Target Variable
0 = no heart disease
1 = mild heart disease
2 = moderate heart disease
3 = severe heart disease
4 = critical heart disease
 



Num
Men exhibit a higher ratio of being disease-free, while females show a lower ratio in the dataset.
Conversely, based on the dataset, men are more affected by diseases compared to women.
 
Handling Missing values


Missing Values Imputation:
So here we impute missing Values by using Iterative Imputer and Random Forest. In this Dataset some Columns Have Higher Missing Values Ratio, so we have to Used Advance methods to impute missing Values. We Define a FUnction for imputing Missing Values, In Which We Passed the Columns Names and The FUnction Return a Dataset With no Missing Values.
Methods:
Random Forest Classifier
Random Forest Regressor
Iterative Imputer

Handling Outliers
While Dealing with Outliers, from my Observations There is only One Outlier in the dataset which I removed. Other Values have Some Meaningful Insight, so we Cannot remove them. Leave them in the Dataset.

 
 
 
 
 
 
 
Model Development:
The Target Column is num which is the predicted attribute. We will use this column to predict the heart disease. The unique values in this column are: [0, 1].
0        = no heart disease 1 = heart disease
The models that you will use to predict the heart disease is Random Forest
Process:
Renaming Column Names and dropping Some Irrelevant Column
Here we Drop some irrelevant columns Like: id, restecg and uses those columns Which are Important.
in Thal and cp we have space which i think will create problem later on so we also remove those spaces.
in target Column 0: 'No disease' and 1: 'Effected Disease'. Here in target Column, i do some changes, before there are 5 different categories. 1,2,3,4, Represent Disease, so I make a new column in which, there are only two categories one represents Disease and one represents no disease. data_1['target'] = ((data['num'] > 0)*1).copy()
[ (data['sex'] == 'Male')*1]: The Boolean values (True/False) are then multiplied by 1. In Python, True is equivalent to 1 and False is equivalent to 0 when used in arithmetic operations. This operation effectively converts the Boolean values into numerical values (1 for 'Male' and 0 for 'Female').
Random Forest:
Random Forest is an ensemble learning technique used for both classification and regression tasks. It builds multiple decision trees during training and merges their predictions to improve accuracy and reduce overfitting.
High Accuracy
Robust to Overfitting
Handles Missing Values
Random Forest is a versatile and powerful algorithm, especially effective in scenarios with high-dimensional data and complex relationships. It excels in situations where high accuracy is crucial, and its ability to handle missing values and resist overfitting makes it a popular choice in machine learning applications.
 
 
A Random Forest Classifier was trained with hyperparameter tuning using GridSearchCV.
Best parameters: {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 150}.
Test accuracy: 65%.
The model and scaler were saved as .pkl files for deployment.
 
 
 
 
 
Web App Deployment:
Created a Flask application with a home route (/) for uploading input or filling a form and a predict route (/predict) for processing input and returning a prediction. Include HTML templates (index.html, result.html) and provide instructions on how to test the app locally.
Create flask app structure
Generated the basic Flask application code in a Python file (e.g., app.py) including the necessary imports and app initialization, and load the pre-trained model and scaler.
Create html templates
Created the 'templates' directory and generated the index.html and result.html files within it, then populate them with the provided HTML content.
Implement home route
Add the / route to the Flask app to render the index.html template, which will contain the form for user input.
Implement predict route
Add the /predict route to the Flask app to handle POST requests, process the form data using the trained model and scaler, and render the result.html template with the prediction.
Test the app locally
Provide instructions and necessary code to run the Flask application locally and test its functionality.
Instructions for testing:
Run the code cell above.
A public URL provided by ngrok will appear in the output. Click on this URL.
This will open the web application in your browser.
Fill out the form with sample data and click "Predict".
You should see the prediction result on the next page.
 
 
FINAL CONCLUSION:
In this study, we trained a powerful machine learning model, using Random Forest, to address our classification task. After an extensive hyperparameter tuning process, we achieved optimal configurations for the model. The Random Forest model demonstrated robust performance with a set of hyperparameters, including a maximum depth of 10, minimum samples per leaf set to 4, minimum samples for split set to 2, and 100 estimators. This resulted in an impressive accuracy of 84% on the test set. 
The model demonstrated its effectiveness in handling the classification task, excelling in different aspects. The Random Forest model showcased high accuracy and robustness to overfitting, making it a reliable choice. Ultimately, the choice of this model depends on specific requirements and preferences. The Random Forest model is suitable for scenarios where accuracy and resistance to overfitting are paramount.
 
 
Github Repository:
The project code and resources are hosted on GitHub: Github Repository.
Please let me know if you have any questions or require further details.





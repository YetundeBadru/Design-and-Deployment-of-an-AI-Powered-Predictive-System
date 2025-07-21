# Design-and-Deployment-of-an-AI-Powered-Predictive-System
Develop and deploy machine learning or AI-based model for a real-world application. Preprocess data, build and evaluate a predictive model, and deploy the solution using Flask so that it can be accessed via a simple web interface.
-----------------------------------------------
Project Objectives
By the end of this project, you will be able to:

• Explore and analyze a real-world dataset (Heart Disease UCI).

• Build and evaluate a supervised ML or AI model.

• Develop a user-facing web interface with Flask.

• Deploy the model for real-time predictions.
---------------------------------------------------------
## Dataset & Problem Definition

Problem: Predict the presence of heart disease from clinical data
Dataset: Heart Disease UCI (https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
Target Variable : num (presence of heart disease)
Features: age, sex, chest pain type, resting blood pressure, cholesterol, etc.
-------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
%matplotlib inline

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
# remove warning
import warnings
warnings.filterwarnings('ignore')

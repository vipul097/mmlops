# Titanic Streamlit App

In this project, we present a solution to the Titanic Competition that achieves a 0.78 score on the test dataset (TOP 13% on 07/09/2023). Moreover, we present a simple but beautiful Streamlit App that allows the users to pretend they are aboard the Titanic and find out their chances of surviving the sinking.
https://www.kaggle.com/code/marcpaulo/titanic-playground-for-new-kagglers-0-78

![demo_screenshot](https://github.com/marcpaulo15/titanic_streamlit/assets/94222660/dd591281-31a3-4885-831c-0068d32361a7)


This project may be exciting for those who are starting their Data Science journey, exploring the Kaggle environment, or want to learn how to use *Streamlit* in their projects. There are also some TO-DO's and other ideas for you to implement in order to further your knowledge and achieve a better score in the competition.

The topics that we cover here are:

**1. Exploratory Data Analysis (EDA)**:
--> Exhaustive feature analysis, outlier detection, missing values, visualize the distribution and structure of the data, initial guess on the importance of each feature, etc. #pandas #seaborn #numpy

**2. Data Preprocessing**:
--> Create a new feature from the existing ones (feature extraction), deal with missing data, design and implement *clean, efficient*, and *reusable* **Pipelines** and Data Transformers using the #sklearn library.

**3. Let's train some Models**:
--> Test basic classification algorithms (LogisticRegression, SVC, RandomForest, GradientBoosting, KNN), and run some **GridSearch** with **Cross-Validation** for **hyperparameter optimization**. Plot the GridSearch results to compare the performance of all the different settings and choose the best configuration.


## Project Structure

```/data/train.csv```: data used to train the models. It contains information about 891 passengers.               
```/images/RMS_titanic.jpg```: images used to decorate the Streamlit app.               
```/main_app/streamlit_app.py```: launches the app using the powerful Streamlit library.             
```/models/trained_grad_boost.pkl```: trained model resulting from the training notebooks and used in the App.                 
```/notebooks/training_playground.ipynb```: long notebook implementing the Data Science part in detail.                 
```/notebooks/training_grad_boost.ipynb```: shot version of the playground notebook in which we only train and save the best model.                    





# Health-Insurance-Cost-Prediction-Web-App
Medical Health Insurance Cost Prediction |TechStack Used: Python, Pandas, Matplotlib, ML Models, GUI, Tkinter, Streamlit |
•	Performed EDA on Kaggle’s Medical cost personal dataset having the features like age, sex, bmi, children, smoker, region & charges.
•	Trained the model using Linear Regression, SVR, Random Forest Regression & Gradient Boosting Regression out of which Gradient Boosting Regression predicts with maximum r2 score of 0.87
•	Saved the model using Joblib, developed the GUI using Tkinter, build and deployed the web application using Streamlit that predicts the insurance cost by entering the features value in web application.

⚙️ How to Run

Clone the repo and open in VS Code / terminal

Create virtual environment and activate it

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

Install required libraries

pip install -r requirements.txt

Start the Streamlit app

streamlit run Insurancecost_streamlit.py

📂 Files

insurance.csv → Dataset

train_model.py → ML model training

train_and_save_model.py → Train + save model

model_joblib_gb.pkl → Trained model

Insurancecost_streamlit.py → Streamlit app UI

requirements.txt → Dependencies

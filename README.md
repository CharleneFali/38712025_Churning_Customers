# 38712025_Churning_Customers
Dataset
The dataset is located at /content/drive/My Drive/Colab Notebooks/CustomerChurn_dataset.csv.
Data Preprocessing
Categorical variables are encoded, and numeric features are scaled.
Model Building
An MLP (Multi-Layer Perceptron) model is created using Keras.
Grid search is performed for hyperparameter tuning.
Model Evaluation
The best model is evaluated on the test set.
Metrics include accuracy and ROC AUC score.
Retraining the Model
The best model is retrained on the entire dataset.
Usage
Open the Jupyter Notebook or Python environment where the code is stored.
Run each cell in sequential order.
Web-based Deployment
A web-based deployment using Streamlit is not included in this code.
File Structure
/content/drive/My Drive/Colab Notebooks/Churn_model.joblib: The saved retrained model.
Dependencies
scikeras
keras
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
Google Colab (if applicable)

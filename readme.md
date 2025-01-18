# Fraud Detection in Financial Transactions

This project focuses on detecting fraudulent financial transactions using machine learning techniques. The dataset used for this project is from [Kaggle](https://www.kaggle.com/datasets/darshandalvi12/fraud-detection-in-financial-transactions).

## Files Overview

1. **cleaning.ipynb**:  
   This Jupyter Notebook contains the data preprocessing and cleaning steps. It includes handling missing values, feature engineering, and applying techniques like SMOTE for dealing with imbalanced datasets.

2. **modelling.ipynb**:  
   This Jupyter Notebook contains the machine learning model building, training, and evaluation. It explores various models, evaluates them using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC, and selects the best performing model. The best performing model is **XGBoost**.

## Installation and Setup

### Prerequisites:
Ensure you have the following libraries installed:

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- xgboost

You can install the required libraries by running the following command:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn xgboost
```

### Dataset

You can download the dataset from Kaggle using the link below:

- [Fraud Detection in Financial Transactions Dataset](https://www.kaggle.com/datasets/darshandalvi12/fraud-detection-in-financial-transactions)

To download the dataset directly:

1. Go to the Kaggle page.
2. Click on the **Download** button to get the dataset as a `.zip` file.
3. Extract the `.zip` file to the project directory.

Once downloaded, ensure the dataset is placed in the same directory as the notebooks or provide the correct path when loading the data.

## How to Use

### Step 1: Data Cleaning (cleaning.ipynb)
1. Open `cleaning.ipynb` in Jupyter Notebook.
2. Load the dataset by running the first few cells.
3. The notebook will guide you through:
   - Handling missing values
   - Encoding categorical features
   - Feature engineering (e.g., transaction time features, amount clustering)
   - Applying SMOTE to balance the dataset

### Step 2: Model Building and Evaluation (modelling.ipynb)
1. Open `modelling.ipynb` in Jupyter Notebook.
2. Load the cleaned and preprocessed data.
3. The notebook will guide you through:
   - Splitting the data into training and test sets
   - Training multiple machine learning models, with **XGBoost** being the best performing model
   - Evaluating the models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC
   - Tuning the model hyperparameters for optimal performance using **XGBoost**

### Step 3: Run the Notebooks
1. Make sure to run both notebooks sequentially (cleaning -> modelling).
2. Evaluate and select the best-performing model based on the evaluation metrics.
3. The **XGBoost** model provides the highest accuracy and best generalization.

## Evaluation Metrics

- **Accuracy**: Overall correctness of the model.
- **Precision**: The proportion of positive predictions that were correct.
- **Recall**: The proportion of actual positives that were correctly identified.
- **F1-Score**: The harmonic mean of precision and recall.

## Results

After running the notebooks, you should have a trained model capable of detecting fraudulent transactions. **XGBoost** has been identified as the best-performing model in this project, achieving superior results compared to other models.

## Conclusion

This project demonstrates a typical approach to fraud detection in financial transactions, utilizing feature engineering and machine learning models to classify transactions as fraudulent or legitimate. By focusing on techniques like handling imbalanced datasets (SMOTE) and using **XGBoost**, the model is well-equipped to handle real-world challenges of imbalanced data.

---

### **Note:**  
- Make sure to configure the dataset path correctly when loading the data into the notebooks.
- Customize the model evaluation depending on your project’s specific goals, especially for fraud detection where **recall** is often prioritized to minimize false negatives.

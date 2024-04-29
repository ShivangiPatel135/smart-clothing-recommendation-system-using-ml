import subprocess

def install_packages(requirements_file):
    # Read the requirements from the file
    with open(requirements_file, 'r') as file:
        requirements = file.readlines()

    # Install each package if not already installed
    for req in requirements:
        package = req.strip()  # Remove leading/trailing whitespace
        try:
            # Check if the package is installed
            subprocess.check_output(['pip', 'show', package])
            print(f"{package} is already installed")
        except subprocess.CalledProcessError:
            # If the package is not installed, install it
            print(f"Installing {package}...")
            subprocess.check_call(['pip', 'install', package])
            print(f"{package} installed successfully")

if __name__ == "__main__":
    requirements_file = 'requirements.txt'
    install_packages(requirements_file)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
# Define the parameter grid to search
def model_running(new_ds,gender_no,cloth,apparel_type,body_type_number,season_number,occasion_number):
  param_grid = {
      'n_estimators': [50, 100, 150],
      'max_depth': [None, 10, 20],
      'min_samples_split': [2, 5, 10]
  }

  # Initialize RandomForestClassifier
  rf_classifier = RandomForestClassifier(random_state=42)
  X = new_ds[['Category','sub-category','Body Type Suitability', 'Season', 'Occasion','Apparel']]
  y = new_ds['Type']

  # Split dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  # Perform Grid Search with Cross-Validation
  grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
  grid_search.fit(X_train, y_train)

  # Train RandomForestClassifier with best parameters
  best_rf_classifier = RandomForestClassifier(**grid_search.best_params_, random_state=42)
  best_rf_classifier.fit(X_train, y_train)

  # Evaluate the model on the test set
  y_pred = best_rf_classifier.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  #print("Model Accuracy:", accuracy)


  # Handle missing values
  new_ds.dropna(inplace=True)

  # Split dataset into features (X) and target variable (y)
  X = new_ds[['Category','sub-category', 'Body Type Suitability', 'Season', 'Occasion','Apparel']]
  y = new_ds['Type']

  # Step 2: Model Training

  # Split dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train Random Forest Classifier
  rf_classifier = RandomForestClassifier(**grid_search.best_params_, random_state=42)
  rf_classifier.fit(X_train, y_train)

  # Step 3: User Preference and Body Type Input
  # Define user input
  user_input = {
      'Category': [gender_no],
      'sub-category':[cloth],
      'Body Type Suitability': [body_type_number],
      'Season': [season_number],
      'Occasion': [occasion_number],
      'Apparel':[apparel_type]
  }

  # Convert user input into DataFrame
  user_df = pd.DataFrame(user_input)

  # Step 4: Prediction
  # Predict 'Type' based on user input
  predicted_type = rf_classifier.predict(user_df)[0]
  print("Predicted Type:", predicted_type)

  # Step 5: Evaluation
  # Evaluate model accuracy on the test set
  y_pred = rf_classifier.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print("Model Accuracy:", accuracy)

  new_ds=new_ds[(new_ds['Type']==int(predicted_type)) ]
  # Split dataset into features (X) and target variable (y)
  X = new_ds[['Type','Category', 'Body Type Suitability', 'Season', 'Occasion','Apparel']]
  y = new_ds['Neckline']

  # Step 2: Model Training

  # Split dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train Random Forest Classifier
  rf_classifier = RandomForestClassifier(**grid_search.best_params_, random_state=42)
  rf_classifier.fit(X_train, y_train)

  # Step 3: User Preference and Body Type Input
  # Define user input
  user_input = {
      'Type':[predicted_type],
      'Category': [gender_no],
      'Body Type Suitability': [body_type_number],
      'Season': [season_number],
      'Occasion': [occasion_number],
      'Apparel':[apparel_type]
  }

  # Convert user input into DataFrame
  user_df = pd.DataFrame(user_input)

  # Step 4: Prediction
  # Predict 'Type' based on user input
  neckline= rf_classifier.predict(user_df)[0]
  print("Predicted neckline:", neckline)
  new_ds=new_ds[(new_ds['Neckline']==int(neckline)) ]

  # Split dataset into features (X) and target variable (y)
  X = new_ds[['Type','Category', 'Body Type Suitability', 'Season', 'Occasion','Neckline','Apparel']]
  y = new_ds['Fit']

  # Split dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train Random Forest Classifier
  rf_classifier = RandomForestClassifier(**grid_search.best_params_, random_state=42)
  rf_classifier.fit(X_train, y_train)

  # Step 3: User Preference and Body Type Input
  # Define user input
  user_input = {
      'Type':[predicted_type],
      'Category': [gender_no],
      'Body Type Suitability': [body_type_number],
      'Season': [season_number],
      'Occasion': [occasion_number],
      'Neckline':[neckline],
      'Apparel':[apparel_type]
  }

  # Convert user input into DataFrame
  user_df = pd.DataFrame(user_input)

  # Step 4: Prediction
  # Predict 'Type' based on user input
  predicted_fit = rf_classifier.predict(user_df)[0]
  print("Predicted Material:", predicted_fit)


  # Split dataset into features (X) and target variable (y)
  X = new_ds[['Type','Fit','Category', 'Body Type Suitability', 'Season', 'Occasion','Neckline','Apparel']]
  y = new_ds['Pattern']

  # Step 2: Model Training

  # Split dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train Random Forest Classifier
  rf_classifier = RandomForestClassifier(**grid_search.best_params_, random_state=42)
  rf_classifier.fit(X_train, y_train)

  # Step 3: User Preference and Body Type Input
  # Define user input
  user_input = {
      'Type':[predicted_type],
      'Fit':[predicted_fit],
      'Category': [gender_no],
      'Body Type Suitability': [body_type_number],
      'Season': [season_number],
      'Occasion': [occasion_number],
      'Neckline':[neckline],
      'Apparel':[apparel_type]
  }

  # Convert user input into DataFrame
  user_df = pd.DataFrame(user_input)

  # Step 4: Prediction
  # Predict 'Type' based on user input
  pattern = rf_classifier.predict(user_df)[0]
  print("Predicted Type:", pattern)


  X = new_ds[['Type','Fit','Pattern','Category', 'Body Type Suitability', 'Season', 'Occasion','Neckline','Apparel']]
  y = new_ds['Material']

  # Step 2: Model Training

  # Split dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train Random Forest Classifier
  rf_classifier = RandomForestClassifier(**grid_search.best_params_, random_state=42)
  rf_classifier.fit(X_train, y_train)

  # Step 3: User Preference and Body Type Input
  # Define user input
  user_input = {
      'Type':[predicted_type],
      'Fit':[predicted_fit],
      'Pattern':[pattern],
      'Category': [gender_no],
      'Body Type Suitability': [body_type_number],
      'Season': [season_number],
      'Occasion': [occasion_number],
      'Neckline':[neckline],
      'Apparel':[apparel_type]
  }

  # Convert user input into DataFrame
  user_df = pd.DataFrame(user_input)

  # Step 4: Prediction
  # Predict 'Type' based on user input
  material = rf_classifier.predict(user_df)[0]
  print("Predicted Type:", material)
  import json

  with open('label_mappings.json', 'r') as file:
      label_mappings = json.load(file)

  for key, value in label_mappings['Type'].items():
      if value == predicted_type:
          print('Type:',key)
          type1=key
  for key, value in label_mappings['Fit'].items():
      if value == predicted_fit:
          print('Fit:',key)
          fit=key
  for key, value in label_mappings['Pattern'].items():
      if value == pattern:
          print('Pattern:',key)
          pattern=key
  for key, value in label_mappings['Material'].items():
      if value == material:
          print('Material:',key)
          material=key
  for key, value in label_mappings['Neckline'].items():
      if value == neckline:
          print('Neckline:',key)
          Neckline=key
  return type1,fit,pattern,material,Neckline
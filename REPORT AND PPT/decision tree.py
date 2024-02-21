import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

path = "C:\\Users\\samue\\Internship4th\\Project_1\\13_GalacticAstraltypeIdentification\\train_dataset.csv"

# Read the data
data = pd.read_csv(path)
print(data.info(), "\n")

# Convert the data into numbers(Encoding)
data['class_n'] = le().fit_transform(data['class'])
print(data, "\n")

# Assuming 'data' is your DataFrame containing the entire dataset
# 'target_column' is the name of the column you're trying to predict

# Separate features (inputs) and target variable (output)
X = data.drop(['class', 'class_n'], axis=1)  # Dropping the column to be predicted
y = data['class_n']  # Only the column to be predicted

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

new_data = pd.DataFrame({
    'alpha': [108.600499600151],
    'delta': [40.2794157558478],
    'u': [17.93123],
    'g': [17.98849],
    'r': [18.37511],
    'i': [18.64214],
    'z': [18.80367],
    'run_ID': [6573],
    'rerun_ID': [301],
    'cam_col': [5],
    'field_ID': [179],
    'spec_obj_ID': [3.313689802502],
    'redshift': [0.0001126093],
    'plate': [2943],
    'MJD': [54502],
    'fiber_ID': [605]
})

# Now, you can use the trained model to make predictions on new data
# For example, assuming 'new_data' is a DataFrame with the same features as your training data
result = model.predict(new_data)
print("Predictions for new data:", result)

if result == 1:
    print("OSO")
elif result == 0:
    print("Galaxy")
elif result == 2:
    print("Star")

# Accuracy of the model
acc = (model.score(X, y)) * 100
print(acc)


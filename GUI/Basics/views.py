from django.shortcuts import render
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def dtml(request):
    if request.method == 'POST':
        # Load and preprocess the dataset
        path = "C:\\Users\\samue\\Internship4th\\Project_1\\13_GalacticAstraltypeIdentification\\train_dataset.csv"
        data = pd.read_csv(path)
        le = LabelEncoder()
        data['class_n'] = le.fit_transform(data['class'])
        
        # Extract form data
        q = float(request.POST.get('q'))
        w = float(request.POST.get('w'))
        e = float(request.POST.get('e'))
        r = float(request.POST.get('r'))
        t = float(request.POST.get('t'))
        m = float(request.POST.get('m'))
        u = float(request.POST.get('u'))
        i = float(request.POST.get('i'))
        o = float(request.POST.get('o'))
        p = float(request.POST.get('p'))
        a = float(request.POST.get('a'))
        s = float(request.POST.get('s'))
        d = float(request.POST.get('d'))
        f = float(request.POST.get('f'))
        g = float(request.POST.get('g'))
        h = float(request.POST.get('h'))

        # Separate features (inputs) and target variable (output)
        X = data.drop(['class','class_n'], axis=1)  # Dropping the column to be predicted
        y = data['class_n']  # Only the column to be predicted

        # Initialize the decision tree classifier
        model = DecisionTreeClassifier()

        # Train the model
        model.fit(X, y)

        # Create new data for prediction
        new_data = pd.DataFrame({
            'alpha': [q],
            'delta': [w],
            'u': [e],
            'g': [r],
            'r': [t],
            'i': [m],
            'z': [u],
            'run_ID': [i],
            'rerun_ID': [o],
            'cam_col': [p],
            'field_ID': [a],
            'spec_obj_ID': [s],
            'redshift': [d],
            'plate': [f],
            'MJD': [g],
            'fiber_ID': [h]
        })

        # Make predictions
        res = model.predict(new_data)

        # Handle prediction results
        result = ''
        if res == 1:
            result = "OSO"
        elif res == 0:
            result = "Galaxy"
        elif res == 2:
            result = "Star"

        # Calculate accuracy
        acc = model.score(X, y) * 100  # Consider using a separate test set for accuracy calculation

        return render(request, 'dtml.html', context={'result': result, 'acc': acc})

    return render(request, 'dtml.html')

        
        


import pandas as pd
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def navie(request):
    if request.method == 'POST':
        # Define the path to the dataset
        path = "C:\\Users\\samue\\Internship4th\\Project_1\\13_GalacticAstraltypeIdentification\\train_dataset.csv"

        # Load and preprocess the dataset
        data = pd.read_csv(path)

        le = LabelEncoder()
        data['class_n'] = le.fit_transform(data['class'])

        # Extract form data
        form_data = {}
        for key in ['q', 'w', 'e', 'r', 't', 'm', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h']:
            form_data[key] = float(request.POST.get(key))

        # Define inputs and outputs
        inputs = data.drop(['class', 'class_n'], axis=1)
        outputs = data['class_n']

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2)

        # Standardize the data
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # Build and train the model
        model = GaussianNB()
        model.fit(x_train, y_train)

        # Predict on test data
        y_pred = model.predict(x_test)

        # Prepare new data for prediction
        new_data = pd.DataFrame([form_data])

        # Transform the new data
        new_data = sc.transform(new_data)

        # Make predictions on new data
        res = model.predict(new_data)
        if res == 0:
            result = 'QSO'
        elif res == 1:
            result = 'GALAXY'
        else:
            result = 'STAR'

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)

        return render(request, 'navie.html', context={'result': result, 'acc': acc})

    return render(request, 'navie.html')

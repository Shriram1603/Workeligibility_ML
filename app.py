from flask import Flask, request, jsonify
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


app = Flask(__name__)

# Your CSV file path
csv_file = 'Workload3.csv'

def add_data_to_csv(data):
    try:
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            model_trainer()
        return True
    except Exception as e:
        return False

# In your Flask route
@app.route('/add_data', methods=['POST'])
def add_data():
    data = request.get_json()  # Assuming you're sending JSON data in the request
    if add_data_to_csv(data):
        return jsonify({"message": "Data added successfully"})
    else:
        return jsonify({"message": "Failed to add data to CSV"}, 500)
    
def model_trainer():
    data = pd.read_csv(csv_file)

    # Encode the "Eligible" column to numeric values (0 for 'no' and 1 for 'yes')
    label_encoder = LabelEncoder()
    data['Skill'] = label_encoder.fit_transform(data['Skill'])
    data.replace({"Eligible":{'no':0,'yes':1}}, inplace=True)
    # Define features and target variable
    X = data.drop('Eligible', axis=1)
    y = data['Eligible']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)







    


if __name__ == '__main__':
    model_trainer()
    app.run(debug=True)
    



# from flask import Flask, request, jsonify, render_template
# import csv
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder

# app = Flask(__name__)

# # Your CSV file path
# csv_file = 'Workload2.csv'

# # Load the trained model (outside of routes for global access)
# def load_trained_model():
#     # Load the dataset
#     df = pd.read_csv(csv_file)

#     # Encode categorical features
#     encoder = LabelEncoder()
#     df['SkillsRequired'] = encoder.fit_transform(df['SkillsRequired'])
#     df.replace({"Status":{'incomplete':0,'completed':1}}, inplace=True)

#     # Define features (X) and the target variable (y)
#     X = df[['SkillsRequired', 'Feedback', 'Complexity', 'Priority', 'Status']]
#     y = df['PersonAssigned']

#     # Create a Random Forest classifier
#     rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

#     # Train the model on the entire dataset
#     rf_classifier.fit(X, y)

#     return rf_classifier

# # Initialize the model
# trained_model = load_trained_model()

# # Add data to CSV file
# def add_data_to_csv(data):
#     try:
#         with open(csv_file, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(data)
#         return True
#     except Exception as e:
#         return False

# # HTML form to input data for prediction
# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         # Get user input from the form
#         skills_required = int(request.form['SkillsRequired'])
#         feedback = int(request.form['Feedback'])
#         complexity = int(request.form['Complexity'])
#         priority = int(request.form['Priority'])
#         status = int(request.form['Status'])

#         # Prepare the input for prediction
#         input_data = [[skills_required, feedback, complexity, priority, status]]

#         # Use the trained model to make a prediction
#         prediction = trained_model.predict(input_data)

#         # Return the prediction result
#         return render_template('prediction_result.html', prediction=prediction[0])

#     return render_template('input_form.html')  # Render the HTML form

# # Add data route
# @app.route('/add_data', methods=['POST'])
# def add_data():
#     data = [int(request.form['SkillsRequired']),
#             int(request.form['Feedback']),
#             int(request.form['Complexity']),
#             int(request.form['Priority']),
#             int(request.form['Status']),
#             int(request.form['PersonAssigned'])]

#     if add_data_to_csv(data):
#         trained_model = load_trained_model()  # Re-train the model with the updated data
#         return render_template('data_added.html', message="Data added successfully and model retrained")

#     return render_template('data_added.html', message="Failed to add data to CSV")

# if __name__ == '__main__':
#     app.run(debug=True)

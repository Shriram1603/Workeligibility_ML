import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv('Workload3.csv')

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

# logistic

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Initialize the Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000)

# Train the model
logistic_regression_model.fit(X_train_vec, y_train)

# Make predictions
y_pred = logistic_regression_model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Save the model to a file
with open('logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(logistic_regression_model, model_file)

# Load the model from the file
with open('logistic_regression_model.pkl', 'rb') as model_file:
    logistic_regression_model = pickle.load(model_file)

# Function to predict emotion from text using Logistic Regression
def predict_emotion_logistic(text):
    text_vec = vectorizer.transform([text])
    prediction = logistic_regression_model.predict(text_vec)
    return prediction[0]

# Define a route for the default URL, which loads the form
@app.route('/')
def form():
    return '''
        <form action="/predict" method="post">
            <label for="text">Enter text:</label>
            <input type="text" id="text" name="text">
            <input type="submit" value="Predict Emotion">
        </form>
    '''

# Define a route for the action of the form, for example '/predict'
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    predicted_emotion = predict_emotion_logistic(text)
    return jsonify({'predicted_emotion': predicted_emotion})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

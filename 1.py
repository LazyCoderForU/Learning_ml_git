import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title of the app
st.title('My First Streamlit App')

# Header
st.header('Welcome to my app')

# Text
st.text('This is a simple Streamlit app.')

# Input box
name = st.text_input('Enter your name:')

# Button
if st.button('Submit'):
    st.write(f'Hello, {name}!')

# Slider
age = st.slider('Select your age:', 0, 100, 25)
st.write(f'Your age is: {age}')

# Checkbox
if st.checkbox('Show/Hide'):
    st.write('Checkbox is checked!')

# Selectbox
option = st.selectbox('Choose an option:', ['Option 1', 'Option 2', 'Option 3'])
st.write(f'You selected: {option}')
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy of the Logistic Regression model: {accuracy:.2f}')

# User input for prediction
st.header('Make a Prediction')
sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, value=0.2)

# Predict button
if st.button('Predict'):
    new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(new_data)
    st.write(f'The predicted class is: {iris.target_names[prediction][0]}')
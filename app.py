from flask import Flask, render_template, request, redirect, url_for, session, jsonify # type: ignore
import random
import numpy as np
import joblib
import re
from flask_sqlalchemy import SQLAlchemy # type: ignore
import os



# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
# app.secret_key = os.urandom(24)
# Configuring the database (SQLite in this case)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Load models and encoders
clf_field = joblib.load("models/field_model.pkl")
clf_degree = joblib.load("models/degree_model.pkl")
clf_career = joblib.load("models/career_model.pkl")

le_interest_1 = joblib.load("models/le_interest_1.pkl")
le_interest_2 = joblib.load("models/le_interest_2.pkl")
le_suggested_field = joblib.load("models/le_suggested_field.pkl")
le_degree_options = joblib.load("models/le_degree_options.pkl")
le_career_paths = joblib.load("models/le_career_paths.pkl")

# User Model for the database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    phone = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

# Create the database tables if they do not exist
with app.app_context():
    db.create_all()

# Mapping first interest to valid second interests
INTEREST_MAP = {
    'Maths': ['Physics', 'Computer', 'Chemistry', 'Biology', 'Finance'],
    'Physics': ['Chemistry', 'Biology', 'Computer', 'Finance', 'Space'],
    'Chemistry': ['Biology', 'Computer', 'Finance', 'Space'],
    'Computer': ['Finance', 'Space'],
    'Kannada': ['History']
}

# Function to predict fields, degree, and career paths
def predict_fields(interest1, interest2):
    """Predict field, degree, and career path based on interests."""
    try:
        # Encode input interests
        interest1_encoded = le_interest_1.transform([interest1])[0]
        interest2_encoded = le_interest_2.transform([interest2])[0]

        # Prepare the input for the model
        input_seq = np.array([interest1_encoded, interest2_encoded], dtype=np.float32).reshape(1, 2)

        # Predict the field, degree, and career path using pre-trained models
        predicted_field_idx = clf_field.predict(input_seq)
        predicted_degree_idx = clf_degree.predict(input_seq)
        predicted_career_idx = clf_career.predict(input_seq)

        # Inverse transform the predicted indices to get the actual labels
        predicted_field = le_suggested_field.inverse_transform([predicted_field_idx])[0]
        predicted_degree = le_degree_options.inverse_transform([predicted_degree_idx])[0]
        predicted_career = le_career_paths.inverse_transform([predicted_career_idx])[0]

        return predicted_field, predicted_degree, predicted_career
    except Exception as e:
        print(f"[Prediction Error] {e}")
        return "Field Prediction Unavailable", "Degree Prediction Unavailable", "Career Prediction Unavailable"

# Home route
@app.route('/')
def home():
    print(f"Session contents: {session}")
    if 'username' not in session:
        print("No username in session. Redirecting to login.")
        return redirect(url_for('login'))
    print(f"User {session.get('username')} is logged in.")
    return render_template('index.html')

# Interest selection route
@app.route('/select_interest', methods=['POST'])
def select_interest():
    interest1 = request.form.get('interest1')
    if not interest1 or interest1 not in INTEREST_MAP:
        return redirect(url_for('home'))

    session['interest1'] = interest1
    second_interests = INTEREST_MAP[interest1].copy()
    random.shuffle(second_interests)

    return render_template('select_second_interest.html', interest1=interest1, second_interests=second_interests)

# Interest validation route
@app.route('/validate-interest', methods=['POST'])
def validate_interest():
    data = request.json
    interest1 = session.get('interest1')
    interest2 = data.get('interest2')

    if not interest1 or not interest2:
        return jsonify({'valid': False, 'message': 'Both interests are required.'}), 400

    valid_choices = INTEREST_MAP.get(interest1, [])
    if interest2 not in valid_choices:
        return jsonify({'valid': False, 'message': 'Invalid second interest selected.'}), 400

    return jsonify({'valid': True})

# Result route
@app.route('/result', methods=['GET'])
def result():
    interest1 = session.get('interest1')
    interest2 = request.args.get('interest2')

    if not interest1 or not interest2:
        return redirect(url_for('home'))

    # Validate the second interest
    if interest2 not in INTEREST_MAP.get(interest1, []):
        return render_template('error.html', message="Invalid second interest selected. Please choose a valid option.")

    # Predict using the trained models
    predicted_field, predicted_degree, predicted_career = predict_fields(interest1, interest2)

    # Prepare suggestions (customize as needed)
    suggestions = {
        'suggested_field': predicted_field,
        'degree_options': predicted_degree,
        'career_paths': predicted_career
    }

    return render_template('result.html', interest1=interest1, interest2=interest2, predicted_career=predicted_career, suggestions=suggestions)

# User Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Validate login credentials (using database)
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['username'] = username  # Store username in session
            print(f"User {username} logged in successfully.")
            return redirect(url_for('home'))
        else:
            print(f"Invalid credentials for user {username}")
            return "Invalid credentials", 401
    
    return render_template('login.html')



# User Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        phone = request.form['phone']
        email = request.form['email']
        password = request.form['password']

        # Basic validation (add more as needed)
        if len(phone) != 10 or not phone.isdigit():
            return "Phone number must be 10 digits", 400
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return "Invalid email format", 400

        # Check if username or email already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "Username already taken", 400

        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            return "Email already registered", 400

        # Save user to database
        new_user = User(username=username, phone=phone, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html')

# User Logout route
@app.route('/logout', methods=['GET'])
def logout():
    session.pop('username', None)
    print("User logged out successfully.")
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
df = pd.read_csv("study_path_suggestions.csv")  # Replace with your actual dataset path

# Encode the categorical features
le_interest_1 = LabelEncoder()
le_interest_2 = LabelEncoder()
le_suggested_field = LabelEncoder()
le_degree_options = LabelEncoder()
le_career_paths = LabelEncoder()

# Encoding the columns
df['interest_1_encoded'] = le_interest_1.fit_transform(df['interest_1'])
df['interest_2_encoded'] = le_interest_2.fit_transform(df['interest_2'])
df['suggested_field_encoded'] = le_suggested_field.fit_transform(df['suggested_field'])
df['degree_options_encoded'] = le_degree_options.fit_transform(df['degree_options'])
df['career_paths_encoded'] = le_career_paths.fit_transform(df['career_paths'])

# Prepare the features (X) and target variables (y)
X = df[['interest_1_encoded', 'interest_2_encoded']]
y_field = df['suggested_field_encoded']
y_degree = df['degree_options_encoded']
y_career = df['career_paths_encoded']

# Train the models using RandomForestClassifier
clf_field = RandomForestClassifier()
clf_degree = RandomForestClassifier()
clf_career = RandomForestClassifier()

clf_field.fit(X, y_field)
clf_degree.fit(X, y_degree)
clf_career.fit(X, y_career)

# Save the trained models and encoders
joblib.dump(clf_field, "models/field_model.pkl")
joblib.dump(clf_degree, "models/degree_model.pkl")
joblib.dump(clf_career, "models/career_model.pkl")

joblib.dump(le_interest_1, "models/le_interest_1.pkl")
joblib.dump(le_interest_2, "models/le_interest_2.pkl")
joblib.dump(le_suggested_field, "models/le_suggested_field.pkl")
joblib.dump(le_degree_options, "models/le_degree_options.pkl")
joblib.dump(le_career_paths, "models/le_career_paths.pkl")

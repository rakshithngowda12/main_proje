import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Flatten
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
df = pd.read_csv("study_path_suggestions.csv")
X = df[['interest_1', 'interest_2', 'preference']]
y_field = df['suggested_field']
y_degree = df['degree_options']
y_career = df['career_paths']

# Encode inputs
le1 = LabelEncoder(); le2 = LabelEncoder(); le3 = LabelEncoder()
X['interest_1'] = le1.fit_transform(X['interest_1'])
X['interest_2'] = le2.fit_transform(X['interest_2'])
X['preference'] = le3.fit_transform(X['preference'])

max_vals = [X[col].max() + 1 for col in X.columns]  # vocab sizes

# Encode outputs
le_field = LabelEncoder(); le_degree = LabelEncoder(); le_career = LabelEncoder()
y_field_enc = le_field.fit_transform(y_field)
y_degree_enc = le_degree.fit_transform(y_degree)
y_career_enc = le_career.fit_transform(y_career)

num_field_classes = len(le_field.classes_)
num_degree_classes = len(le_degree.classes_)
num_career_classes = len(le_career.classes_)

# Convert X to sequence shape
X_seq = X.values.reshape((X.shape[0], 3))  # (samples, timesteps)

# Build model
input_layer = Input(shape=(3,))
embedding = Embedding(input_dim=max(max_vals), output_dim=8)(input_layer)
lstm = LSTM(32)(embedding)
field_out = Dense(num_field_classes, activation='softmax', name='field')(lstm)
degree_out = Dense(num_degree_classes, activation='softmax', name='degree')(lstm)
career_out = Dense(num_career_classes, activation='softmax', name='career')(lstm)

model = Model(inputs=input_layer, outputs=[field_out, degree_out, career_out])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_seq, [to_categorical(y_field_enc), to_categorical(y_degree_enc), to_categorical(y_career_enc)], epochs=1000, batch_size=8, verbose=1)

# Save everything
model.save("lstm_model.h5")
joblib.dump((le1, le2, le3, le_field, le_degree, le_career), "lstm_encoders.pkl")

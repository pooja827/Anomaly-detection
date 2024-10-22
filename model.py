import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('water_potability.csv')

# Convert date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.date
# Define feature columns
feature_columns = ['ph', 'solids', 'hardness', 'conductivity', 'organic_carbon', 'turbidity']

# Impute missing values using KNN
imputer = KNNImputer(n_neighbors=5)
df[feature_columns] = imputer.fit_transform(df[feature_columns])

# Normalize features
scaler = MinMaxScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# Prepare data
X = df[feature_columns].values
y = df['Potability'].values  # Ensure 'Potability' is your target column

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Reshape for LSTM (samples, timesteps, features)
timesteps = 1  # Adjust this as needed
X_resampled = X_resampled.reshape((X_resampled.shape[0], timesteps, X_resampled.shape[1]))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.5))  # Increased dropout
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(Dropout(0.5))  # Increased dropout
model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.5))  # Increased dropout
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, 
                    validation_split=0.2)

# Evaluate the model
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary

# Classification report and confusion matrix
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f'ROC AUC Score: {roc_auc:.2f}')

# Visualize training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
# Save the trained model
model.save('water_potability_model.h5')

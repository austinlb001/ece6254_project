# This file takes in the encoder.csv data set and runs a neural network with a softmax and outputs a 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

df = pd.read_csv('encoded.csv')

# encoded.csv only has number of goals from each team, this function determines the result of the game from that
def match_result(row):
    if row['home_team_goal'] > row['away_team_goal']:
        return 'Home_win'
    elif row['home_team_goal'] < row['away_team_goal']:
        return 'Away_win'
    else:
        return 'Draw'

df['Result'] = df.apply(match_result, axis=1)

label_encoder = LabelEncoder()
df['Result'] = label_encoder.fit_transform(df['Result'])

X = df.drop(['Result', 'home_team_goal', 'away_team_goal'], axis=1)
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feedforward NN, testing out different things (regularizer, dropout, activations)
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax') # Softmax for 3 classes: "Home Win" "Away Win" and "Draw"
])

optimizer = Adam(learning_rate=0.01) # Adjust for tunning
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)


# Testing - Getting around ~40% accuracy currently
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
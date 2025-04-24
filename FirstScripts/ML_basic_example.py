import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

# Create random dataset
np.random.seed(42)
X = np.random.rand(1000, 10)  # 1000 samples, 10 features
y = (X.sum(axis=1) > 5).astype(int)  # Binary target based on sum of features

# Convert to pandas DataFrame
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
df['target'] = y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- TensorFlow Model ----------------------
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tf_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# ---------------------- PyTorch Model ----------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

# Prepare PyTorch datasets
torch_X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
torch_y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(torch_X_train)
    loss = criterion(outputs, torch_y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

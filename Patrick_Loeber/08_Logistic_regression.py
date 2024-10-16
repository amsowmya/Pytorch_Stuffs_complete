import torch 
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


bc = load_breast_cancer()
print(bc.keys())

X = bc.data
y = bc.target

n_samples, n_features = X.shape
print(n_samples, n_features)

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    

model = LogisticRegression(n_features)

learning_rate = 0.01
epochs = 100

criterian = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    y_predicted = model(X_train)

    loss = criterian(y_predicted, y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f"Epoch : {epoch+1}, loss = {loss.item():.4f}")

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_num = y_predicted.round()

    acc = y_predicted_num.eq(y_test).sum() / X_test.shape[0]

    print(f"Test accuracy is : {acc:.4f}")


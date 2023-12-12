from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Train a model using KNN
def train_knn(X_train, y_train):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    return knn

# Evaluate the model on a dataset
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy
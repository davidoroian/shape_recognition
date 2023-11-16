from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Train a model using SVM
def train_svm(X_train, y_train):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    return svm

# Evaluate the model on a dataset
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy
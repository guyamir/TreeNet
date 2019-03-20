from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from loader import load
from sklearn.tree import DecisionTreeClassifier

l = load('wine', 1,0.2)
X_train, X_valid, y_train, y_valid = l.X_train, l.X_valid, l.y_train, l.y_valid

# Create decision tree classifier object
decisiontree = DecisionTreeClassifier()

# Train model
model = decisiontree.fit(X_train, y_train)

po = model.predict(X_valid)

acc_count = 0
for i in range(len(po)):
    if po[i] == y_valid[i]:
        acc_count += 1
print(acc_count/len(po))

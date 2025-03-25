import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# dataset
data = pd.DataFrame({
    'Age': [25, 45, 35, 23, 52, 40, 60, 30, 27, 50],
    'Income': [30_000, 80_000, 60_000, 25_000, 90_000, 70_000, 120_000, 45_000, 35_000, 85_000],
    'PreviousPurchases': [1, 5, 2, 0, 7, 3, 10, 1, 1, 6],
    'WillBuy': [0, 1, 1, 0, 1, 1, 1, 0, 0, 1]
})

# Features and labels
X = data[['Age', 'Income', 'PreviousPurchases']]
y = data['WillBuy']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# First tree to graph
tree = model.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=['Age', 'Income', 'PreviousPurchases'],
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True
)
plt.show()
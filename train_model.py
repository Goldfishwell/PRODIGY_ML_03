import pickle as pk
import random
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

with open('data1.pickle', 'rb') as f:
    data = pk.load(f)

random.shuffle(data)

X = [i[0] for i in data]
y = [i[1] for i in data]


xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.10, random_state=42)
model = make_pipeline(StandardScaler(),LinearSVC(C=1.0, max_iter=5000))
model.fit(xtrain, ytrain)

with open('model.sav', 'wb') as f:
    pk.dump(model, f)

print("Model trained")

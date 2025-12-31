import pickle as pk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
categories = ['Cat', 'Dog']

with open('model.sav', 'rb') as f:
    model = pk.load(f)

img_path = r'C:\Users\Umesh\Downloads\Prodigy tasks\task3\test\7.jpg'              #Custom image


with open('data1.pickle', 'rb') as f:
    data = pk.load(f)

img2 = cv2.imread(img_path)
img = cv2.imread(img_path,0)
img = cv2.resize(img,(50,50))
img_flat = img.flatten()

X = [i[0] for i in data]
y = [i[1] for i in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
prediction = model.predict([img_flat])
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy *100,' %')
if(prediction == 0):
    print("This image is a Cat")
elif(prediction == 1):
    print("This image is a Dog")
else:
    print('error')

plt.imshow(img2)
plt.title(f"Prediction: {categories[prediction[0]]}")
plt.axis('off')
plt.show()

from cgi import test
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')["labels"]

print(pd.Series(y).value_counts())

classes = ["A","B","C","D","E","F","G","H","I","J","K","L"
            ,"M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 3000, train_size= 7000, random_state= 9)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

classifier = LogisticRegression(solver = 'saga', multi_class= 'multinomial').fit(X_train_scaled, y_train)

def getPrediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized  = image_bw.resize((28,28), Image.ANTIALIAS)

    pixel_filter = 20

    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scale = np.clip(image_bw_resized - min_pixel,0,255)
    max_pixel = np.max(image_bw_resized)

    image_bw_resized_inverted_scale = np.asarray(image_bw_resized_inverted_scale)/max_pixel

    test_sample = np.array(image_bw_resized_inverted_scale).reshape(1,784)
    test_pred = classifier.predict(test_sample)

    return test_pred[0]
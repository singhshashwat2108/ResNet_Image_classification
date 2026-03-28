import matplotlib.pyplot as plt
import numpy as np
from utils import load_data

x_train, x_test, y_train, y_test = load_data()

class_names = ['airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck']

# Sample images
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')

plt.suptitle("Sample Images from CIFAR-10")
plt.show()

# Class distribution
labels = y_train.flatten()
plt.hist(labels, bins=10)
plt.title("Class Distribution")
plt.show()

# Pixel distribution
plt.hist(x_train.flatten(), bins=50)
plt.title("Pixel Intensity Distribution")
plt.show()
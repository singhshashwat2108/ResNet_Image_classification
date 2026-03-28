import matplotlib.pyplot as plt
import pickle

# OPTIONAL: Save histories from each model and compare

# Example placeholder (manually copy accuracy values if needed)
baseline_acc = [0.5, 0.6, 0.65]
advanced_acc = [0.6, 0.7, 0.75]
resnet_acc = [0.65, 0.75, 0.8]

plt.plot(baseline_acc, label='Baseline CNN')
plt.plot(advanced_acc, label='Advanced CNN')
plt.plot(resnet_acc, label='ResNet')

plt.title("Model Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
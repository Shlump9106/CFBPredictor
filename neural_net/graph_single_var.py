import pandas as pd
import matplotlib.pyplot as plt

# Load the training results CSV
df = pd.read_csv('training_results.csv')

# Plot accuracy over epochs
plt.figure()
plt.plot(df['epoch'], df['roc_auc'])
plt.xlabel('Epoch')
plt.ylabel('ROC AUC')
plt.title('ROC AUC over Epochs')
plt.show()

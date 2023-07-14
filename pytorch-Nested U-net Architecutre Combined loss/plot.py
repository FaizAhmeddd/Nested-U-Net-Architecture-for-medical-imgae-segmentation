import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('log.csv')

# Plot IoU
plt.plot(df['epoch'], df['iou'], label='IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.show()

# Plot Dice
plt.plot(df['epoch'], df['dice'], label='Dice')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()
plt.show()

# Plot Accuracy
plt.plot(df['epoch'], df['acc'], label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Precision
plt.plot(df['epoch'], df['prec'], label='Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Plot Recall
plt.plot(df['epoch'], df['rec'], label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.show()

# Plot F1 score
plt.plot(df['epoch'], df['f1'], label='F1 score')
plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.legend()
plt.show()

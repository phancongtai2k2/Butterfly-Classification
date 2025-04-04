import pandas as pd
import matplotlib.pyplot as plt

history_df = pd.read_csv('D:\\download\\Butterfly-Classification-main\\training_history.csv')

plt.figure(figsize=(12, 6))

# Vẽ biểu đồ Loss
plt.subplot(1, 2, 1)
plt.plot(history_df['Epoch'], history_df['Training Loss'], label='Training Loss', color='blue')
plt.plot(history_df['Epoch'], history_df['Validation Loss'], label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

# Vẽ biểu đồ Accuracy
plt.subplot(1, 2, 2)
plt.plot(history_df['Epoch'], history_df['Training Accuracy'], label='Training Accuracy', color='purple')
plt.plot(history_df['Epoch'], history_df['Validation Accuracy'], label='Validation Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

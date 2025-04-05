import os
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

root_dir = "D:\download\Butterfly-Classification-main\dataset\dataset"  

class_counts = defaultdict(int)

for split in ['train', 'val', 'test']:
    split_path = os.path.join(root_dir, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            image_count = len([
                f for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
            ])
            class_counts[class_name] += image_count

# Chuyển dict thành 2 list để vẽ
classes = list(class_counts.keys())
counts = list(class_counts.values())

plt.figure(figsize=(18, 6))
sns.barplot(x=classes, y=counts, palette="viridis")

plt.xticks(rotation=90)
plt.title("Distribution of Butterfly Classes")
plt.xlabel("Butterfly Classes")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()

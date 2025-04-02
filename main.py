# main.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import create_model
from PIL import Image
import os

# Hàm dự đoán với mô hình
def predict_image(image_path, model, device):
    # Đọc và xử lý ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Thêm batch dimension

    # Tiến hành dự đoán
    model.eval()  # Chuyển mô hình về chế độ đánh giá
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()  # Trả về lớp dự đoán

def load_model(model_path, device):
    model = create_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def main():
    # Đường dẫn đến mô hình đã huấn luyện
    model_path = 'best_model.pth'
    
    # Kiểm tra GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Tải mô hình
    model = load_model(model_path, device)

    # Đường dẫn ảnh cần dự đoán
    image_path = r"C:\Users\phant\Desktop\data_buom\Chequered Skipper1.jpg"  # Thay đổi đường dẫn đến ảnh của bạn

    # Dự đoán kết quả
    predicted_class = predict_image(image_path, model, device)
    print(f'Predicted class: {predicted_class}')

if __name__ == '__main__':
    main()

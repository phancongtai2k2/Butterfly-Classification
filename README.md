# Butterfly-Classification with CNN Model

Dự án này xây dựng và thử nghiệm mô hình Mạng Nơ-ron Chuyển Convolutional (CNN) để phân loại hình ảnh vào một trong 75 lớp. Mô hình được huấn luyện, đánh giá và có thể dự đoán trên các hình ảnh mới.

**Yêu cầu**
Trước khi chạy dự án, hãy đảm bảo rằng bạn đã cài đặt các thư viện sau:
  Python 3.7 hoặc cao hơn
  
  PyTorch
  
  torchvision
  
  scikit-learn
  
  matplotlib
  
  seaborn
 
  PIL (Pillow)

Bạn có thể cài đặt các thư viện cần thiết bằng lệnh sau:    pip install torch torchvision scikit-learn matplotlib seaborn pillow

**Cấu trúc dự án**
  main.py: Điểm vào chính để phân loại hình ảnh. Tải mô hình đã huấn luyện và thực hiện dự đoán trên hình ảnh mới.
  
  model.py: Định nghĩa kiến trúc mô hình CNN.
  
  train.py: Mã nguồn huấn luyện mô hình sử dụng bộ dữ liệu.
  
  test.py: Kiểm tra mô hình đã huấn luyện trên bộ dữ liệu kiểm tra và tạo các chỉ số đánh giá hiệu suất (độ chính xác, báo cáo phân loại, ma trận nhầm lẫn).

**Các bước thực hiện dự án**
_Bước 1: Chuẩn bị bộ dữ liệu_
  Đảm bảo rằng bạn có bộ dữ liệu với các hình ảnh được sắp xếp vào các thư mục sau:
    dataset/
      train/        chứa các hình ảnh dùng để huấn luyện.
      val/          chứa các hình ảnh dùng để xác thực.
      test/         chứa các hình ảnh dùng để kiểm tra.
_Bước 2: Huấn luyện mô hình_
  Để huấn luyện mô hình, chạy train.py. Mã nguồn này sẽ:
    Tải bộ dữ liệu.
    Định nghĩa mô hình CNN.
    Huấn luyện mô hình trên bộ dữ liệu huấn luyện.
    Đánh giá mô hình trên bộ dữ liệu xác thực.
    Lưu mô hình tốt nhất dưới tên best_model.pth.
_Bước 3: Kiểm tra mô hình_
  Sau khi mô hình được huấn luyện, bạn có thể kiểm tra hiệu suất của nó bằng cách sử dụng test.py. Mã nguồn này sẽ tải mô hình đã huấn luyện và đánh giá nó trên bộ dữ liệu kiểm tra. Nó cũng sẽ tạo ra:
  Độ chính xác
  Báo cáo phân loại
  Ma trận nhầm lẫn (lưu dưới dạng confusion_matrix.png)
_Bước 4: Dự đoán trên hình ảnh_
  Để dự đoán lớp của một hình ảnh mới bằng mô hình đã huấn luyện, sử dụng main.py. Mã nguồn này sẽ:
  Tải mô hình đã huấn luyện.
  Đọc hình ảnh đầu vào.
  Thực hiện dự đoán.

**by_TCP ngày 4/2/2025**
    

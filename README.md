#Phát hiện sinh viên có khả năng bỏ học bằng các thuật toán Machine Learning

#Bài toán
- Bài toán phát hiện sinh viên có khả năng bỏ học nhằm góp phần giảm thiểu tình trạng bỏ học và trượt đại học, bằng cách sử dụng các kỹ thuật học máy để xác định sinh viên có nguy cơ ngay từ giai đoạn đầu trên con đường học vấn, từ đó có thể đưa ra các chiến lược hỗ trợ phù hợp. Bộ dữ liệu bao gồm thông tin được biết đến tại thời điểm sinh viên nhập học – con đường học vấn, nhân khẩu học và các yếu tố kinh tế xã hội. Bài toán được xây dựng dưới dạng một bài toán phân loại ba loại (bỏ học, đã nhập học và tốt nghiệp) vào cuối thời lượng khóa học thông thường,bao gồm:

- Bài toán được mô hình hóa dưới dạng phân loại đa lớp với ba nhãn mục tiêu:
Dropout (bỏ học)
Enrolled (đang theo học)
Graduate (tốt nghiệp)

#Mục tiêu
Xây dựng hệ thống Machine Learning dự đoán
- Dropout: Bỏ học
- Enrolled: đang theo học
- Graduate: tốt nghiệp

#Dataset
Nguồn: Kaggle
https://www.kaggle.com/code/subhajeetdas/student-dropout-prediction<img width="1473" height="105" alt="image" src="https://github.com/user-attachments/assets/5a3e900d-1628-46ac-ba41-6bcec2ca3d28" />

#Thuộc tính dữ liệu
o	 Marital status: Tình trạng hôn nhân của sinh viên
o	 Nacionality: Quốc tịch của sinh viên
o	 Gender: Giới tính của sinh viên
o	 Age at enrollment: Tuổi của sinh viên tại thời điểm nhập học
o	 Mother's qualification: Trình độ học vấn của mẹ
o	 Father's qualification: Trình độ học vấn của cha
o	 Mother's occupation: Nghề nghiệp của mẹ
o	 Father's occupation: Nghề nghiệp của cha
o	 Application mode: Phương thức nộp hồ sơ
o	 Application order: Thứ tự nộp hồ sơ
o	 Course: Ngành sinh viên viên đăng ký
o	 Previous qualification: Bằng cấp hoặc trình độ học vấn trước khi vào đại học
o	 Previous qualification (grade): điểm của trình độ học vấn trước khi sinh viên nhập học đại học
o	 Daytime/evening attendance: Sinh viên học ban ngày hay buổi tối
o	 International: Có phải sinh viên quốc tế hay không
o	 Displaced: Sinh viên có phải người di dời / tị nạn hay không
o	 Educational special needs: Sinh viên có nhu cầu giáo dục đặc biệt hay không
o	 Debtor: Sinh viên có nợ học phí hay không
o	 Tuition fees up to date: Sinh viên có đóng học phí đúng hạn hay không
o	 Scholarship holder: Sinh viên có nhận học bổng hay không
o	 Unemployment rate: Tỷ lệ thất nghiệp (thời điểm nhập học)
o	 Inflation rate: Tỷ lệ lạm phát
o	 GDP: Tổng sản phẩm quốc nội
o	 Curricular units 1st sem (credited): Số học phần được công nhận
o	 Curricular units 1st sem (enrolled): Số học phần đăng ký
o	 Curricular units 1st sem (evaluations): Số học phần được đánh giá
o	 Curricular units 1st sem (approved): Số học phần đạt
o	 Curricular units 1st sem (grade): Điểm trung bình học kỳ 1
o	 Curricular units 1st sem (without evaluations): Số học phần không tham gia đánh giá
o	 Curricular units 2nd sem (credited): Số học phần được công nhận
o	 Curricular units 2nd sem (enrolled): Số học phần đăng ký
o	 Curricular units 2nd sem (evaluations): Số học phần được đánh giá
o	 Curricular units 2nd sem (approved): Số học phần đạt
o	 Curricular units 2nd sem (grade): Điểm trung bình học kỳ 2
o	 Curricular units 2nd sem (without evaluations): Số học phần không tham gia đánh giá
o	Target:
  - Dropout: Bỏ học
  - Enrolled: Đang theo học
  - Graduate: Đã tốt nghiệp

#Pipeline
Dataset → EDA → Clean → Encode → Normalize → Train → Evaluate → Inference

##Các bước chính
Xử lý missing values
Xử lý giá trị không đồng nhất
StandardScaler 
OneHotEncoder
GridSearchCV để tối ưu tham số
Train & Evaluate

#Mô hình
- Logistic Regression
- Support Vector Machine
- Decision Tree
- Random Forest
- Gradient Boosting

#Kết quả
- Đối với bài toán phân loại đa lớp
| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|---------:|----------:|-------:|---------:|
| Logistic Regression | 0.595    | 0.56      | 0.58   | 0.57     |
| SVM                 | 0.603    | 0.56      | 0.60   | 0.58     |
| Decision Tree       | 0.598    | 0.50      | 0.70   | 0.58     |
| Random Forest       | 0.595    | 0.55      | 0.58   | 0.57     |
| Gradient boosting   | 0.595    | 0.55      | 0.61   | 0.58     |

- Đối với bài toán phân loại nhị phân khi gộp 2 lớp Graduate và Enrolled thành lớp 0 và Drop out thành lớp 1
| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|---------:|----------:|-------:|---------:|
| Logistic Regression | 0.685    | 0.51      | 0.75   | 0.61     |
| SVM                 | 0.684    | 0.51      | 0.76   | 0.61     |
| Decision Tree       | 0.558    | 0.41      | 0.88   | 0.56     |
| Random Forest       | 0.693    | 0.52      | 0.74   | 0.61     |
| Gradient boosting   | 0.714    | 0.55      | 0.59   | 0.57     |

#Cách chạy
- Bước 1: Mở môi trường thực hành code. Gợi ý nên sử dụng Google Colab
- Bước 2: Nhập lệnh !git clone https://github.com/NguyenDoKhaiHoan/machinelearning.git để tài các mục trong git hub
- Bước 3: Nhấn đúp lần lượt các mục như app để tải notbook về sau đó có thể import và chạy lần lượt các cell code
  + Với data có thể làm theo hướng dẫn để tải dữ liệu về
  + Đối với VScode nếu chưa có các thư viện cần thiết thì nên lần lượt dùng lệnh "pip install thư viên" để cài đặt
  + Đối với Google Colab hay các môi trường có sẵn ở trên mạng thì chỉ cần chạy lần lượt các cell code để có kết quả giống như trong notebook
  + Với mục demo khuyến nghị chạy code đến cell chứa phần import joblib để tải các file các cột đặc trưng chính và best_model rồi dùng VScode gõ lệnh python app.py để chạy ra demo
    


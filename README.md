# Phát hiện sinh viên có khả năng bỏ học bằng các thuật toán Machine Learning

## Bài toán
Bài toán phát hiện sinh viên có khả năng bỏ học nhằm góp phần giảm thiểu tình trạng bỏ học và trượt đại học. Mục tiêu là sử dụng các kỹ thuật học máy để xác định sớm sinh viên có nguy cơ ngay từ giai đoạn đầu (tại thời điểm nhập học), từ đó nhà trường có thể đưa ra các chiến lược hỗ trợ phù hợp.

Bộ dữ liệu bao gồm thông tin được biết đến tại thời điểm sinh viên nhập học:
- Con đường học vấn
- Nhân khẩu học
- Các yếu tố kinh tế xã hội

Bài toán được mô hình hóa dưới dạng **phân loại đa lớp** với ba nhãn mục tiêu:
- **Dropout**: Bỏ học  
- **Enrolled**: Đang theo học  
- **Graduate**: Tốt nghiệp  

Ngoài ra, bài toán cũng được chuyển đổi sang **phân loại nhị phân** để tập trung vào mục tiêu phát hiện nguy cơ bỏ học:
- Lớp `0`: Graduate + Enrolled  
- Lớp `1`: Dropout  
---

## Mục tiêu xây dựng hệ thống dự đoán
Xây dựng hệ thống Machine Learning dự đoán trạng thái của sinh viên:
- **Dropout**: Bỏ học  
- **Enrolled**: Đang theo học  
- **Graduate**: Đã tốt nghiệp  

---

## Dataset
Nguồn dữ liệu: Kaggle
```text
https://www.kaggle.com/code/subhajeetdas/student-dropout-prediction
```

---

## Thuộc tính dữ liệu

### Thông tin cá nhân & nhân khẩu học
- **Marital status**: Tình trạng hôn nhân của sinh viên  
- **Nacionality**: Quốc tịch của sinh viên  
- **Gender**: Giới tính của sinh viên  
- **Age at enrollment**: Tuổi của sinh viên tại thời điểm nhập học  

### Thông tin gia đình
- **Mother's qualification**: Trình độ học vấn của mẹ  
- **Father's qualification**: Trình độ học vấn của cha  
- **Mother's occupation**: Nghề nghiệp của mẹ  
- **Father's occupation**: Nghề nghiệp của cha  

### Thông tin tuyển sinh & học tập
- **Application mode**: Phương thức nộp hồ sơ  
- **Application order**: Thứ tự nộp hồ sơ  
- **Course**: Ngành sinh viên đăng ký  
- **Previous qualification**: Trình độ học vấn trước khi vào đại học  
- **Previous qualification (grade)**: Điểm của trình độ học vấn trước khi nhập học đại học  
- **Daytime/evening attendance**: Sinh viên học ban ngày hay buổi tối  
- **International**: Có phải sinh viên quốc tế hay không  

### Tình trạng kinh tế – xã hội & hỗ trợ
- **Displaced**: Sinh viên có phải người di dời / tị nạn hay không  
- **Educational special needs**: Sinh viên có nhu cầu giáo dục đặc biệt hay không  
- **Debtor**: Sinh viên có nợ học phí hay không  
- **Tuition fees up to date**: Sinh viên có đóng học phí đúng hạn hay không  
- **Scholarship holder**: Sinh viên có nhận học bổng hay không  

### Chỉ số kinh tế vĩ mô (tại thời điểm nhập học)
- **Unemployment rate**: Tỷ lệ thất nghiệp (thời điểm nhập học)  
- **Inflation rate**: Tỷ lệ lạm phát  
- **GDP**: Tổng sản phẩm quốc nội  

### Kết quả học tập theo học kỳ (Curricular units)
- **Curricular units 1st sem (credited)**: Số học phần được công nhận  
- **Curricular units 1st sem (enrolled)**: Số học phần đăng ký  
- **Curricular units 1st sem (evaluations)**: Số học phần được đánh giá  
- **Curricular units 1st sem (approved)**: Số học phần đạt  
- **Curricular units 1st sem (grade)**: Điểm trung bình học kỳ 1  
- **Curricular units 1st sem (without evaluations)**: Số học phần không tham gia đánh giá  

- **Curricular units 2nd sem (credited)**: Số học phần được công nhận  
- **Curricular units 2nd sem (enrolled)**: Số học phần đăng ký  
- **Curricular units 2nd sem (evaluations)**: Số học phần được đánh giá  
- **Curricular units 2nd sem (approved)**: Số học phần đạt  
- **Curricular units 2nd sem (grade)**: Điểm trung bình học kỳ 2  
- **Curricular units 2nd sem (without evaluations)**: Số học phần không tham gia đánh giá  

### Nhãn mục tiêu (Target)
- **Dropout**: Bỏ học  
- **Enrolled**: Đang theo học  
- **Graduate**: Đã tốt nghiệp  

---

## Pipeline
**Dataset → EDA → Clean → Encode → Normalize → Train → Evaluate → Inference**

### Các bước chính
- Xử lý **missing values**
- Xử lý **giá trị không đồng nhất**
- Chuẩn hóa dữ liệu bằng **StandardScaler**
- Mã hóa dữ liệu bằng **OneHotEncoder**
- Tối ưu tham số bằng **GridSearchCV**
- **Train & Evaluate** mô hình

---

## Mô hình
- Logistic Regression
- Support Vector Machine
- Decision Tree
- Random Forest
- Gradient Boosting

---

## Kết quả

### 1) Bài toán phân loại đa lớp
| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|---------:|----------:|-------:|---------:|
| Logistic Regression | 0.595    | 0.56      | 0.58   | 0.57     |
| SVM                 | 0.603    | 0.56      | 0.60   | 0.58     |
| Decision Tree       | 0.598    | 0.50      | 0.70   | 0.58     |
| Random Forest       | 0.595    | 0.55      | 0.58   | 0.57     |
| Gradient boosting   | 0.595    | 0.55      | 0.61   | 0.58     |

### 2) Bài toán phân loại nhị phân
Gộp **Graduate** và **Enrolled** thành lớp `0`, và **Dropout** thành lớp `1`.

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|---------:|----------:|-------:|---------:|
| Logistic Regression | 0.685    | 0.51      | 0.75   | 0.61     |
| SVM                 | 0.684    | 0.51      | 0.76   | 0.61     |
| Decision Tree       | 0.558    | 0.41      | 0.88   | 0.56     |
| Random Forest       | 0.693    | 0.52      | 0.74   | 0.61     |
| Gradient boosting   | 0.714    | 0.55      | 0.59   | 0.57     |

---

## Cách chạy

### Chạy trên Google Colab hoặc VScode (khuyến nghị chạy trên Google Colab)
**Bước 1:** Mở Google Colab hoặc VScode  
**Bước 2:** Clone source về môi trường chạy code:
```bash
!git clone https://github.com/NguyenDoKhaiHoan/machinelearning.git
```
**Bước 3:** Nhấn đúp lần lượt các mục như app để tải notebook về sau đó có thể import và chạy lần lượt các cell code
  + Với data có thể làm theo hướng dẫn để tải dữ liệu về
  + Đối với VScode nếu chưa có các thư viện cần thiết thì nên lần lượt dùng lệnh "pip install thư viên" để cài đặt
  + Đối với Google Colab hay các môi trường có sẵn ở trên mạng thì chỉ cần chạy lần lượt các cell code để có kết quả giống như trong notebook
  + Với mục demo khuyến nghị chạy code đến cell chứa phần import joblib để tải các file các cột đặc trưng chính và best_model rồi dùng VScode gõ lệnh python app.py để chạy được giao diện demo phát hiện sinh viên có nguy cơ bỏ học
---
## Tác giả
Họ và tên: Nguyễn Đỗ Khải Hoàn
Mã lớp: 12423TN
Mã sinh viên: 12423012


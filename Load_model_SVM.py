# import the necessary packages
from preprocessing import imagetoarraypreprocessor
from preprocessing import simplepreprocessor
from datasets import simpledatasetloader
from imutils import paths
import numpy as np
import cv2
import pickle

# Khởi tạo danh sách nhãn
classLabels = ["cat", "dog", "panda"]

#Lấy danh sách các hình ảnh trong tập dữ liệu sau đó lấy mẫu ngẫu nhiên
# ảnh theo chỉ số idx để đưa vào đường dẫn hình ảnh
print("[INFO] Đang nạp ảnh mẫu để phân lớp...")
imagePaths = np.array(list(paths.list_images("image"))) # xác định số file trong dataset (folder image)
idxs = np.random.randint(0, len(imagePaths), size=(10,)) # Trả về 10 idxs ngẫu nhiên
imagePaths = imagePaths[idxs]

# Tiền xử lý dữ liệu ảnh
sp = simplepreprocessor.SimplePreprocessor(32, 32) # Thiết lập kích thước ảnh 32 x 32
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor() # Gọi hàm để chuyển ảnh sang mảng


# Nạp dataset ảnh từ đĩa lư vào đối tượng data
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.reshape((data.shape[0], 3072))

# Nạp model (network) đã được train (pre-trained)
print("[INFO] Nạp model mạng pre-trained ...")
model = pickle.load(open('svmpickle_file.pkl', 'rb'))

# Dự đoán nhãn (label) ảnh đầu vào. Ảnh được lưu trong data
print("[INFO] Đang dự đoán để phân lớp...")
preds = model.predict(data)

# Lặp qua tất cả các file ảnh trong imagePaths và thực hiện trên từng ảnh, gồm:
# Nạp ảnh --> tạo label dự đoán trên ảnh --> Hiển thị ảnh
for (i, imagePath) in enumerate(imagePaths):
    # Nạp ảnh
    image = cv2.imread(imagePath)
    # Tạo label dự đoán trên ảnh
    cv2.putText(image, "label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Hiển thị ảnh
    cv2.imshow("Image", image)
    cv2.waitKey(0)








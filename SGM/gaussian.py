import cv2
import numpy as np

# 读取原图
image = cv2.imread('./img/image.png')

# 降采样到 640x480
image_downsampled = cv2.resize(image, (224, 224))

# 上采样回原始尺寸
image_upsampled = cv2.resize(image_downsampled, (image.shape[1], image.shape[0]))

# 保存结果图像
cv2.imwrite('./imgdownsampled_image.jpg', image_downsampled)
cv2.imwrite('./imgupsampled_image.jpg', image_upsampled)


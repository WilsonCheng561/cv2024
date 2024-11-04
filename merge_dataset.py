import os
import shutil

# 定义目标路径
gesture_recognition_dir = 'gesture_recongnition/train_gesture_data'
numbers_asl_dir = 'Numbers_ASL'

# 确保 gesture_recognition/train_gesture_data 中的 0-9 文件夹存在
for i in range(10):
    class_dir = os.path.join(gesture_recognition_dir, str(i))
    os.makedirs(class_dir, exist_ok=True)

# 遍历 Numbers_ASL 中的每个 numbers_* 文件夹
for numbers_folder in os.listdir(numbers_asl_dir):
    numbers_folder_path = os.path.join(numbers_asl_dir, numbers_folder)

    # 检查是否为文件夹
    if os.path.isdir(numbers_folder_path):
        # 遍历 0-9 文件夹
        for class_folder in range(10):
            class_folder_path = os.path.join(numbers_folder_path, str(class_folder))

            # 检查 0-9 文件夹是否存在
            if os.path.isdir(class_folder_path):
                # 获取 class_folder_path 中的所有图片文件
                for img_file in os.listdir(class_folder_path):
                    img_path = os.path.join(class_folder_path, img_file)

                    # 确保文件是图片（可以扩展为检查文件扩展名）
                    if os.path.isfile(img_path):
                        # 新文件名，避免重复。使用 numbers_folder 和原文件名进行唯一标识
                        new_img_name = f"{numbers_folder}_{img_file}"
                        target_path = os.path.join(gesture_recognition_dir, str(class_folder), new_img_name)

                        # 复制图片到目标文件夹
                        shutil.copy(img_path, target_path)
                        print(f"Copied {img_path} to {target_path}")

print("Data merging completed.")

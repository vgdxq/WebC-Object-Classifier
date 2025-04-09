import os
from PIL import Image


def check_and_clean_images(directory):
    problematic_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Перевіряє, чи це валідне зображення
                img.close()
            except Exception as e:
                print(f"Problem with file {file_path}: {e}")
                problematic_files.append(file_path)

    # Видалення проблемних файлів (розкоментуйте, якщо хочете видалити)
    for file_path in problematic_files:
        os.remove(file_path)
        print(f"Removed: {file_path}")


TRAIN_DIR = '../data/train'
VAL_DIR = '../data/validation'
check_and_clean_images(TRAIN_DIR)
check_and_clean_images(VAL_DIR)
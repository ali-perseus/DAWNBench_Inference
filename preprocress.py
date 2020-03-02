import os
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms


def get_paths_and_labels(base_folder):
    path_list = []
    label_names = os.listdir(base_folder)
    for i, sub_folder in enumerate(label_names):
        file_folder = base_folder + sub_folder + "/"
        for f_name in os.listdir(file_folder):
            #skip xml
            if 'xml' not in f_name:
                path_list.append((file_folder + f_name, i))
    return path_list


if __name__ == "__main__":
    transform_func = transforms.Compose([
        transforms.Resize(256, Image.BICUBIC),
        transforms.CenterCrop((224, 224)),
    ])
    path_to_dataset = "/path/to/validation/"
    path_to_image_cache_dir = "image/"
    path_to_label = "label.txt"
    #create image cache dir
    if not os.path.exists(path_to_image_cache_dir):
        os.mkdir(path_to_image_cache_dir)
    with open(path_to_label, "w") as f_write:
        for i, (f_path,
                label_id) in enumerate(get_paths_and_labels(path_to_dataset)):
            img_pil = Image.open(f_path).convert("RGB")
            img_pil = transform_func(img_pil)
            f_name = f_path.split("/")[-1]
            f_name_type = f_path.split(".")[-1]
            f_name = f_name.replace(f_name_type, "png")
            #write label file
            new_path = path_to_image_cache_dir + f_name
            f_write.write(new_path + f" {label_id}\n")
            #write resize file
            #img_pil.save(new_path)
            #to skip the libpng error
            #we use opencv to write the resize&crop image
            img_data = np.array(img_pil)
            cv2.imwrite(new_path, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))
            if i % 100 == 0:
                print(f"----{i+1}----finished-----")
        print(f"convert {i+1} images")
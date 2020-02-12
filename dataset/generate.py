import argparse
import os
import cv2


def construct_dataset(src, class_A_path, class_R_path):
    for (root, dirs, files) in os.walk(src):
        file_cnt = 1
        for dir in dirs:
            if dir == "final_dataset":
                continue

            dir_path = os.path.join(src, dir)
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)

                label = dir.split("-")[1]
                ext = file.split(".")[1]
                new_file_name = str(file_cnt) + "_" + label + "." + ext

                img = cv2.imread(file_path)
                if label == "A":
                    cv2.imwrite(os.path.join(class_A_path, new_file_name), img)
                elif label == "R":
                    cv2.imwrite(os.path.join(class_R_path, new_file_name), img)

                file_cnt += 1


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--src", required=True, help="Path to src data directory")
ap.add_argument("-d", "--dst", required=True, help="Path to dst data directory")
args = vars(ap.parse_args())

src_dir_path = args["src"]
dst_dir_path = args["dst"]

if not os.path.exists(dst_dir_path):
    try:
        os.mkdir(dst_dir_path)

        class_A_path = os.path.join(dst_dir_path, "A")
        os.mkdir(class_A_path)

        class_R_path = os.path.join(dst_dir_path, "R")
        os.mkdir(class_R_path)
    except OSError:
        print("Failed while trying to create one of the specified directories")
    else:
        print("Successfully created all of the specified directories ... ")

    construct_dataset(src_dir_path, class_A_path, class_R_path)

    print(len(os.listdir(class_A_path)))
    print(len(os.listdir(class_R_path)))
# encoding=utf-8

import os
import gzip
import shutil


def ungzip_files(gz_folder, extract_folder):
    """
    批量解压文件夹中的所有.gz压缩包到指定路径
    """
    os.makedirs(extract_folder, exist_ok=True)
    for file_name in os.listdir(gz_folder):
        file_path = os.path.join(gz_folder, file_name)
        if file_name.endswith('.gz'):
            extract_file_name = file_name[:-3]
            extract_file_path = os.path.join(extract_folder, extract_file_name)
            with gzip.open(file_path, 'rb') as f_in:
                with open(extract_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"已解压: {file_name} -> {extract_file_path}")
        else:
            print(f"跳过非.gz文件: {file_name}")


if __name__ == "__main__":
    gz_folder = input("请输入包含.gz压缩包的文件夹路径: ").strip('"')
    extract_folder = input("请输入解压后的文件存放路径: ").strip('"')
    ungzip_files(gz_folder, extract_folder)
    print("所有.gz压缩包解压完成！")

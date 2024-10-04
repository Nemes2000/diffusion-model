import zipfile
import shutil
import argparse

def load_partition_description(file_name: str):
    with open(file_name) as input:
        return [line.strip().split(' ') for line in input.readlines()]
    
def split_file_names_into_train_test_validation(partitions: list[list[str, str]]):
    train_test_validation_map: dict[str, list[str]] = {}
    for partition in partitions:
        if train_test_validation_map.get(partition[1]) is None: train_test_validation_map[partition[1]] = []
        train_test_validation_map[partition[1]].append(partition[0])

    return train_test_validation_map

def unzip_image_zip(file_name: str, save_folder: str): 
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(save_folder)

def split_images_into_train_test_validation(image_folder: str, dataset_folder: str, train_test_val_map: dict[str, list[str]]):
    sub_folders = ['train', 'validation', 'test']

    for key, value in train_test_val_map.items():
        for image in value:
            shutil.copyfile(f'{image_folder}/{image}', f'{dataset_folder}/{sub_folders[int(key)]}/{image}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('partition-description', type=str)
    parser.add_argument('--unzip-filename', type=str)
    parser.add_argument('--remove-image-folder')
    parser.add_argument('workdir', type=str)
    parser.add_argument('image-folder', type=str)

    workdir = parser.workdir
    dataset_folder = f'{workdir}/dataset'
    image_folder = f'{workdir}/{parser.image_folder}'
    partitions = load_partition_description(f'{workdir}/{parser.partition_description}')
    train_test_val_map = split_file_names_into_train_test_validation(partitions)

    if parser.unzip_filename:
        unzip_image_zip(f'{workdir}/{parser.unzip_filename}', workdir)

    split_images_into_train_test_validation(image_folder, dataset_folder, train_test_val_map)

    if parser.remove_image_folder:
        shutil.rmtree(image_folder, ignore_errors=True)
    









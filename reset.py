import os
import shutil

def delete_all_folders(path):
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)

if __name__ == '__main__':
    path = 'node_data/'
    delete_all_folders(path)
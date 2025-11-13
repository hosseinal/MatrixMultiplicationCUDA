import os
import glob

def write_mtx_files_to_txt(directory):
    with open('mat_list_all.txt', 'w') as output_file:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".mtx"):
                    file_path = os.path.join(root, file)
                    file_path = file_path[1:]
                    output_file.write(file_path)
                    output_file.write('\n')

directory = '.'  # Current directory
write_mtx_files_to_txt(directory)

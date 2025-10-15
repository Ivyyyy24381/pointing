import os
import argparse

def rename_metadata_files(folder):
    files = sorted([
        f for f in os.listdir(folder)
        if f.startswith('_Color_metadata_') and f.endswith('.txt')
    ])

    for i, filename in enumerate(files, start=1):
        new_name = f'_Color_metadata_{i:04}.txt'
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        print(f'Renamed: {filename} -> {new_name}')
    files = sorted([
        f for f in os.listdir(folder)
        if f.startswith('_Depth_metadata_') and f.endswith('.txt')
    ])

    for i, filename in enumerate(files, start=1):
        new_name = f'_Depth_metadata_{i:04}.txt'
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        print(f'Renamed: {filename} -> {new_name}')


def rename_color_files(folder, file_type = '.png'):
    files = sorted([
        f for f in os.listdir(folder)
        if f.startswith('_Color_') and f.endswith(file_type)
    ])

    for i, filename in enumerate(files, start=1):
        new_name = f'_Color_{i:04}{file_type}'
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        print(f'Renamed: {filename} -> {new_name}')

def rename_depth_files(folder, file_type = '.png'):
    files = sorted([
        f for f in os.listdir(folder)
        if f.startswith('_Depth_') and f.endswith(file_type)
    ])

    for i, filename in enumerate(files, start=1):
        new_name = f'_Depth_{i:04}{file_type}'
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        print(f'Renamed: {filename} -> {new_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rename metadata files with sequential numbers.")
    parser.add_argument('folder', type=str, help='Path to the folder containing the metadata files')

    args = parser.parse_args()
    path = args.folder
    # Get the last directory name
    last_dir = os.path.basename(os.path.normpath(path))

    rename_metadata_files(path)

    if last_dir.lower() == 'color':
        print("This is a Color folder.")

        rename_color_files(args.folder, file_type = '.png')
        rename_depth_files(args.folder, file_type = '.png')

    elif last_dir.lower() == 'depth':
        print("This is a Depth folder.")

        rename_color_files(args.folder, file_type = '.raw')
        rename_depth_files(args.folder, file_type = '.raw')

        

# /home/xhe71/Desktop/dog_data/BDL244_Hannah/1/Color/
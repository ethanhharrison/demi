import os

def rename_files_in_order(folder_path: str):
    """
    Renames all files in the given folder sequentially: 1.ext, 2.ext, ...
    """
    files = sorted(
        [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    )

    for i, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{i}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_name}")

# Example:
# rename_files_in_order("/path/to/your/folder")
if __name__ == '__main__':
    rename_files_in_order("./images/positives")
    rename_files_in_order("./images/negatives")
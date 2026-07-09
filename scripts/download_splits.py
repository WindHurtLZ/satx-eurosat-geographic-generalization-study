import os
import urllib.request

def download_splits():
    # This script lives in scripts/, so the project root is one directory up
    # and the data directory is <root>/data.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_dir = os.path.join(base_dir, "data", "splits_data")
    os.makedirs(splits_dir, exist_ok=True)
    
    splits = ["train", "val", "test"]
    print(f"Downloading splits to: {splits_dir}")
    for split in splits:
        # 1. Download Spatial Split
        filename_spatial = f"eurosat-spatial-{split}.txt"
        file_path_spatial = os.path.join(splits_dir, filename_spatial)
        if not os.path.exists(file_path_spatial):
            url = f"https://huggingface.co/datasets/torchgeo/eurosat/raw/main/{filename_spatial}"
            print(f"Downloading {filename_spatial}...")
            with urllib.request.urlopen(url) as response, open(file_path_spatial, 'wb') as out_file:
                out_file.write(response.read())

        # 2. Download Standard Split
        filename_std = f"eurosat-{split}.txt"
        file_path_std = os.path.join(splits_dir, filename_std)
        if not os.path.exists(file_path_std):
            url = f"https://huggingface.co/datasets/torchgeo/eurosat/raw/main/{filename_std}"
            print(f"Downloading {filename_std}...")
            with urllib.request.urlopen(url) as response, open(file_path_std, 'wb') as out_file:
                out_file.write(response.read())

    print("All splits downloaded successfully!")

if __name__ == "__main__":
    download_splits()

import kagglehub

def get_list(list="datasets/downloaded.txt"):
    with open(list, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def add_to_list(item, list="datasets/downloaded.txt"):
    with open(list, "a") as f:
        f.write(item + "\n")

def download_dataset(path):
    datasets = get_list()
    try:    
        if path in datasets:
            return "Dataset already downloaded"
        else:
            kagglehub.dataset_download(path)
            add_to_list(path)
            return "Dataset downloaded"
    except Exception as e:
        return f"Error downloading dataset {path}: {e}"
    
def update_datasets(list="datasets/dataset_library.txt"):
    datasets = get_list(list)
    size = len(datasets)
    for i, dataset in enumerate(datasets):
        try:
            kagglehub.dataset_download(dataset)
            print(f"{i}/{size} - {dataset} downloaded")
        except Exception as e:
            print(f"Error downloading dataset {dataset}: {e}")
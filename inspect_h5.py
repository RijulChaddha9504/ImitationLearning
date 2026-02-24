import h5py

def print_hdf5_item(name, obj):
    # Print the name and whether it is a Dataset (array) or Group (folder)
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name} | shape: {obj.shape} | type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group:   {name}")

# Open the file in read-only mode
file_path = "demonstrations/robomimic_dataset.hdf5"
print(f"--- Inspecting {file_path} ---")

with h5py.File(file_path, "r") as f:
    # visititems recursively goes through all groups and datasets in the file
    f.visititems(print_hdf5_item)

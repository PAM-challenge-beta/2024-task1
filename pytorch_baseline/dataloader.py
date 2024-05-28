import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, table_path, num_classes):
        self.hdf5_path = hdf5_path
        self.table_path = table_path + "/data"
        self.num_classes = num_classes

        self._hf = h5py.File(self.hdf5_path, 'r')
        
        with h5py.File(self.hdf5_path, 'r') as file:
            dataset = file[self.table_path]
            self.length = dataset.shape[0]
            self.indices = list(range(dataset.shape[0]))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset = self._hf[self.table_path]
        data = dataset[self.indices[idx]]['data']
        label = dataset[self.indices[idx]]['label']

        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return data, label
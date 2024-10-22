import torch
import numpy as np

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *arrays, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        self.arrays = arrays

        self.dataset_len = self.arrays[0].shape[0]
        self.batch_size = batch_size if batch_size < self.dataset_len else self.dataset_len
        self.shuffle = shuffle
        self.data = arrays[0]
        self.targets = arrays[1]

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = np.random.permutation(self.dataset_len)
            self.arrays = [t[r] for t in self.arrays]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        data = self.data[self.i:self.i+self.batch_size]
        targets = self.targets[self.i:self.i+self.batch_size]
        data, targets = torch.tensor(data), torch.tensor(targets)
        self.i += self.batch_size
        return data, targets

    def __len__(self):
        return self.n_batches

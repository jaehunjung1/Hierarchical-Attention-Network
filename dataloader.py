from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from dataset import collate_fn


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        self.n_samples = len(dataset)

        self.sampler = RandomSampler(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'pin_memory': True,
            'collate_fn': collate_fn,
            'shuffle': False,  # must be false to use sampler
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)
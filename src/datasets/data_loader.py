from torch.utils.data import Dataset, DataLoader
from .dataset_factory import *

conf = conf
dataset = get_detset(conf)


data_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=2)

for i, batch in enumerate(data_loader, 0):
    print(i, batch)
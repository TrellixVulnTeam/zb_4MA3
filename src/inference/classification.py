# coding=utf-8
import argparse
import torch
# from visdom import Visdom
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, '.')
# print(sys.path)

torch.multiprocessing.set_sharing_strategy('file_system')

from transforms.transform_factory import get_all_transform
from datasets.dataset_factory import get_dataset
from config.parser import Config
from models.model_factory import get_classification_model


"""
    @author:luckygong
"""

# logger
# to txt & screen
logger = Logger()

# parse config
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='', help='config path')
args = parser.parse_args()

conf = Config(args.config_path)


# load data
data_transform = get_all_transform(conf)
dataset = get_dataset(conf, data_transform)
input = DataLoader(dataset=dataset, batch_size=100, shuffle=conf.is_train, num_workers=10)
# logger.debug('the data number is %d' % )

# define model
model = get_classification_model(conf)


# forward
for i, (batch_x, batch_y) in enumerate(list(input)):
    predict_i = model(batch_x).argmax(dim=1)
    num_correct = torch.eq(predict_i, batch_y).sum().float().item()
    print('Inference batch %d done, right number is %d, batch number is %d' % (i + 1, num_correct, len(batch_x)))

# metric
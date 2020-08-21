# coding=utf-8

import configparser
import os


class Config():
    def __init__(self, path):
        cf = configparser.ConfigParser()
        cf.read(path)
        # for level_one_key in self.cf:
        #     for level_two_key in self.cf[level_one_key]):
        #         setattr(self, self.cf[level_one_key][level_two_key])

        # global
        self.is_train = cf.get("global", "is_train")
        self.is_train = True if self.is_train == 'yes' else False

        # model
        self.model_name = cf.get("model", "model_name")
        self.pretrained_model_path_or_url = cf.get('model', 'pretrained_model_path_or_url')
        self.drop_rate = float(cf.get('model', 'drop_rate'))

        # data
        self.dataset_name = cf.get("data", "dataset_name")
        self.data_classes = int(cf.get("data", "data_classes"))
        self.dataset_local_path = cf.get("data", "dataset_local_path")
        self.dataset_is_download = cf.get("data", "dataset_is_download")
        self.dataset_is_download = True if self.dataset_is_download == 'yes' else False
        assert os.path.isdir(self.dataset_local_path) or self.dataset_is_download, 'dataset path not exist'

        # for cifar nothing
        # for Folder
        if(self.dataset_name == 'imagefolder' or self.dataset_name == 'datasetfolder'):
            self.loader = cf.get("data", "loader")
            if(self.dataset_name == 'datasetfolder'):
                self.extensions = cf.get("data", "extensions")
                if(self.extensions == ''):
                    self.extensions = None
            self.is_valid_file = cf.get("data", "is_valid_file") # ‘’
            if(self.is_valid_file == ''):
                self.is_valid_file = None
                self.loader = None


        # for ImageNet
        if(self.dataset_name == 'imagenet'):
            self.split = cf.get("data", "split")
        # for QMNIST
        if(self.dataset_name == 'qmnist'):
            self.what = cf.get("data", "what")
            self.compat = cf.get("data", "compat")

        assert os.path.isdir(self.dataset_local_path) or self.dataset_is_download, 'dataset path not exist'

        # transform
        self.dataset_transform = {}
        self.target_transform = None
        for k,v in cf['transform'].items():
            if(v != ''):
                self.dataset_transform[k] = eval(v)
            else:
                self.dataset_transform[k] = None



        self.print()

    def print(self):
        message = ''
        message += '---------------------------------- Options --------------------------------\n'
        for k, v in vars(self).items():
            # print(k, v)
            message += '{:>30}: {:<30}\n'.format(str(k), str(v))
        message += '---------------------------------- End ------------------------------------'
        print(message)


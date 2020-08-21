from . import *

def get_dataset(conf, transform = None, transforms = None, target_transform = None):
    #cifar
    if (conf.dataset_name == 'cifar10'):
        return CIFAR10(root=conf.dataset_local_path,
                       train=conf.is_train,
                       transform=transform,
                       target_transform=target_transform,
                       download=conf.dataset_is_download)
    elif (conf.dataset_name == 'cifar100'):
        return CIFAR100(root=conf.dataset_local_path,
                        train=conf.is_train,
                        transform=transform,
                        target_transform=target_transform,
                        download=conf.dataset_is_download)
    #MNIST
    elif (conf.dataset_name == 'emnist'):
        return EMNIST(root=conf.dataset_local_path,
                      split=conf.split,
                      train=conf.is_train,
                      transform=transform,
                      target_transform=target_transform,
                      download=conf.dataset_is_download)
    elif (conf.dataset_name == 'fashionmnist'):
        return FashionMNIST(root=conf.dataset_local_path,
                            train=conf.is_train,
                            target_transform=target_transform,
                            transform=transform,
                            download=conf.dataset_is_download)
    elif (conf.dataset_name == 'kmnist'):
        return KMNIST(root=conf.dataset_local_path,
                      train=conf.is_train,
                      target_transform=target_transform,
                      transform=transform,
                      download=conf.dataset_is_download)
    elif (conf.dataset_name == 'mnist'):
        return MNIST(root=conf.dataset_local_path,
                     train=conf.is_train,
                     target_transform=target_transform,
                     transform=transform,
                     download=conf.dataset_is_download)
    elif (conf.dataset_name == 'qmnist'):
        return QMNIST(root=conf.dataset_local_path,
                      train=conf.is_train,
                      what=conf.what,
                      compat=conf.compat,
                      target_transform=target_transform,
                      transform=transform,
                      download=conf.dataset_is_download)
    # Folder
    elif (conf.dataset_name == 'datasetfolder'):
        return DatasetFolder(root=conf.dataset_local_path,
                             loader=conf.loader,
                             extensions=conf.extensions,
                             transform=transform,
                             target_transform=target_transform,
                             is_valid_file=conf.is_valid_file)
    elif (conf.dataset_name == 'imagefolder'):
        return ImageFolder(root=conf.dataset_local_path,
                           loader=conf.loader,
                           transform=transform,
                           target_transform=target_transform,
                           is_valid_file=conf.is_valid_file
                           )

    # ImageNet
    elif (conf.dataset_name == 'imagenet'):
        return ImageNet(root=conf.dataset_local_path,
                        split=conf.split,
                        target_transform=target_transform,
                        transform=transform,
                        loader=conf.loader)

    #
    elif(conf.dataset_name == 'Caltech101'):
        return Caltech101(root=conf.dataset_local_path, target_type=conf.target_type, transform=transform,
                       target_transform=target_transform, dataset_is_download=conf.dataset_is_download)
    elif(conf.dataset_name == 'Caltech256'):
        return Caltech256(root=conf.dataset_local_path, transform=transform,
                       target_transform=target_transform, dataset_is_download=conf.dataset_is_download)

    #
    elif(conf.dataset_name == 'CelebA'):
        return CelebA(root=conf.dataset_local_path, split=conf.split, target_type=conf.target_type,
                        transform=transform,target_transform=target_transform,
                        dataset_is_download=conf.dataset_is_download)

    #
    elif (conf.dataset_name == 'Cityscapes'):
        return Cityscapes(root=conf.dataset_local_path, split=conf.split, target_type=conf.target_type,
                        transform=transform, target_transform=target_transform,
                        transforms=transforms)

    #
    elif (conf.dataset_name == 'CocoCaptions'):
        return CocoCaptions(root=conf.dataset_local_path, annFile=conf.annFile,
                        transform=transform, target_transform=target_transform,
                        transforms=transforms)
    elif (conf.dataset_name == 'CocoDaptions'):
        return CocoDetection(root=conf.dataset_local_path, annFile=conf.annFile,
                        transform=transform, target_transform=target_transform,
                        transforms=transforms)

    #
    elif (conf.dataset_name == 'FakeData'):
        return FakeData(size=conf.size, image_size=conf.image_size,num_classes=conf.num_classes,
                        transform=transform, target_transform=target_transform,
                        random_offset=conf.random_offset)

    #
    elif (conf.dataset_name == 'Flickr8k'):
        return Flickr8k(root=conf.dataset_local_path, ann_file=conf.ann_file,
                        transform=transform, target_transform=target_transform)
    elif (conf.dataset_name == 'Flickr30k'):
        return Flickr30k(root=conf.dataset_local_path, ann_file=conf.ann_file,
                        transform=transform, target_transform=target_transform)


    #
    elif (conf.dataset_name == 'HMDB51'):
        return HMDB51(root=conf.dataset_local_path, frames_per_clip=conf.frames_per_clip, step_between_clips=conf.step_between_clips,
                        transform=transform, fold=conf.fold, train=conf.is_train,
                        is_valid_file=conf.is_valid_file)


    #
    elif (conf.dataset_name == 'Kinetics400'):
        return Kinetics400(root=conf.dataset_local_path, frames_per_clip=conf.frames_per_clip,
                        step_between_clips=conf.step_between_clips,
                        transform=transform,)

    #
    elif (conf.dataset_name == 'LSUN'):
        return LSUN(root=conf.dataset_local_path, classes=conf.classes,
                        target_transform=target_transform,
                        transform=transform,)



    #
    elif (conf.dataset_name == 'Omniglot'):
        return Omniglot(root=conf.dataset_local_path, background=conf.background,
                        target_transform=target_transform,
                        transform=transform,
                        dataset_is_download=conf.dataset_is_download)

    #
    elif (conf.dataset_name == 'PhotoTour'):
        return PhotoTour(root=conf.dataset_local_path, name=conf.name,
                        transform=transform,
                        dataset_is_download=conf.dataset_is_download)

    #
    elif (conf.dataset_name == 'SBDataset'):
        return SBDataset(root=conf.dataset_local_path, image_set=conf.image_set, mode=conf.mode,
                        transform=transform,
                        dataset_is_download=conf.dataset_is_download)

    #
    elif (conf.dataset_name == 'SBU'):
        return SBU(root=conf.dataset_local_path, target_transform=target_transform,
                        transform=transform,
                        dataset_is_download=conf.dataset_is_download)

    #
    elif (conf.dataset_name == 'SEMEION'):
        return SEMEION(root=conf.dataset_local_path, target_transform=target_transform,
                        transform=transform,
                        dataset_is_download=conf.dataset_is_download)

    #
    elif (conf.dataset_name == 'STL10'):
        return STL10(root=conf.dataset_local_path, split=conf.split, folds=conf.folds,
                       target_transform=target_transform, transform=transform,
                        dataset_is_download=conf.dataset_is_download)

    #
    elif (conf.dataset_name == 'SVHN'):
        return SVHN(root=conf.dataset_local_path, split=conf.split,
                     transform=transform,target_transform=target_transform,
                    dataset_is_download=conf.dataset_is_download)

    #
    elif (conf.dataset_name == 'UCF101'):
        return UCF101(root=conf.dataset_local_path, annotation_path=conf.annotation_path,frames_per_clip=conf.frames_per_clip,
                      step_between_clips=conf.step_between_clips,fold=conf.fold,
                      transform=transform,
                      train=conf.is_train)

    #
    elif (conf.dataset_name == 'USPS'):
        return USPS(root=conf.dataset_local_path,
                      target_transform=target_transform,
                      dataset_is_download=conf.dataset_is_download,
                      transform=transform,
                      train=conf.is_train)

    #
    elif (conf.dataset_name == 'VOCDetection'):
        return VOCDetection(root=conf.dataset_local_path,
                            year=conf.year,
                            image_set=conf.image_set,
                            transform=transform,
                            target_transform=target_transform,
                            transforms=transforms,
                            dataset_is_download=conf.dataset_is_download)
    elif (conf.dataset_name == 'VOCSegmentation'):
        return VOCSegmentation(root=conf.dataset_local_path,
                            year=conf.year,
                            image_set=conf.image_set,
                            transform=transform,
                            target_transform=target_transform,
                            transforms=transforms,
                            dataset_is_download=conf.dataset_is_download)
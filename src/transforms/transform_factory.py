from . import *

def get_single_transform(name, **kwargs):
    if(name == 'totensor'):
        return ToTensor()
    elif(name == 'topilimage'):
        return ToPILImage(**kwargs)
    elif (name == 'normalize'):
        return Normalize(**kwargs)
    elif (name == 'resize'):
        return Resize(**kwargs)
    elif(name == 'centercrop'):
        return CenterCrop(**kwargs)
    elif(name == 'randomcrop'):
        return RandomCrop(**kwargs)
    elif(name == 'randomhorizontalflip'):
        return RandomHorizontalFlip(**kwargs)
    elif(name == 'randomverticalflip'):
        return RandomVerticalFlip(**kwargs)
    elif(name == 'randomresizedcrop'):
        return RandomResizedCrop(**kwargs)
    elif(name == 'lineartransformation'):
        return LinearTransformation(**kwargs)
    elif(name == 'colorjitter'):
        return ColorJitter(**kwargs)
    elif(name == 'randomrotation'):
        return RandomRotation(**kwargs)
    elif(name == 'grayscale'):
        return Grayscale(**kwargs)

def get_all_transform(conf):
    all_transform = []
    for transform in conf.dataset_transform:
        if(conf.dataset_transform[transform] != None):
            all_transform.append(get_single_transform(transform, **conf.dataset_transform[transform]))
    return Compose(all_transform)
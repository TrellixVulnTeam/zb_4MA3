# coding=utf-8

"""
    @author:luckygong
"""

from . import *

def get_classification_model(conf):
    if(conf.model_name == 'alexnet'):
        return alexnet(pretrained=conf.pretrained_model_path_or_url, num_classes = conf.data_classes)
    elif(conf.model_name == 'googlenet'):
        return googlenet(pretrained=conf.pretrained_model_path_or_url, num_classes = conf.data_classes,
                         aux_logits = False, transform_input = False, init_weights = None, blocks=None)  # TODO add conf
    elif (conf.model_name == 'vgg11'):
        return vgg11(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif (conf.model_name == 'vgg11_bn'):
        return vgg11_bn(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif (conf.model_name == 'vgg13'):
        return vgg13(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif (conf.model_name == 'vgg13_bn'):
        return vgg13_bn(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif (conf.model_name == 'vgg16'):
        return vgg16(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif (conf.model_name == 'vgg16_bn'):
        return vgg16_bn(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif (conf.model_name == 'vgg19'):
        return vgg19(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif (conf.model_name == 'vgg19_bn'):
        return vgg19_bn(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif(conf.model_name == 'densenet121'):
        return densenet121(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes, drop_rate=conf.drop_rate,
                           bn_size=4, memory_efficient=False)  # TODO add conf
    elif(conf.model_name == 'densenet161'):
        return densenet161(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes, drop_rate=conf.drop_rate,
                           bn_size=4, memory_efficient=False)  # TODO add conf
    elif(conf.model_name == 'densenet169'):
        return densenet169(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes, drop_rate=conf.drop_rate,
                           bn_size=4, memory_efficient=False)  # TODO add conf
    elif(conf.model_name == 'densenet201'):
        return densenet201(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes, drop_rate=conf.drop_rate,
                           bn_size=4, memory_efficient=False)  # TODO add conf
    elif(conf.model_name == 'inceptionv3'):
        return inception_v3(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes,
                            aux_logits=True, transform_input=False, inception_blocks=None, init_weights=None)  # TODO add conf
    elif(conf.model_name == 'resnet18'):
        return resnet18(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes,
                        zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None)  # TODO add conf
    elif(conf.model_name == 'resnet34'):
        return resnet34(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes,
                        zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None)  # TODO add conf
    elif(conf.model_name == 'resnet50'):
        return resnet50(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes,
                        zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None)  # TODO add conf
    elif(conf.model_name == 'resnet101'):
        return resnet101(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes,
                        zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None)  # TODO add conf
    elif(conf.model_name == 'resnet152'):
        return resnet152(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes,
                        zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None)  # TODO add conf
    elif(conf.model_name == 'resnext50_32x4d'):
        return resnext50_32x4d(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes,
                        zero_init_residual=False, replace_stride_with_dilation=None, norm_layer=None)  # TODO add conf
    elif(conf.model_name == 'resnext101_32x8d'):
        return resnext101_32x8d(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes,
                        zero_init_residual=False, replace_stride_with_dilation=None, norm_layer=None)  # TODO add conf
    elif(conf.model_name == 'wide_resnet50_2'):
        return wide_resnet50_2(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes,
                        zero_init_residual=False, groups=1, replace_stride_with_dilation=None, norm_layer=None)  # TODO add conf
    elif(conf.model_name == 'wide_resnet101_2'):
        return wide_resnet101_2(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes,
                        zero_init_residual=False, groups=1, replace_stride_with_dilation=None, norm_layer=None)  # TODO add conf
    elif(conf.model_name == 'mobilenet_v2'):
        return mobilenet_v2(pretrained = conf.pretrained_model_path_or_url, num_classes=conf.data_classes,
                 width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None, norm_layer=None)  # TODO add conf
    elif(conf.model_name == 'squeezenet1_0'):
        return squeezenet1_0(pretrained = conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif (conf.model_name == 'squeezenet1_1'):
        return squeezenet1_1(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif(conf.model_name == 'shufflenet_v2_x0_5'):
        return shufflenet_v2_x0_5(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif (conf.model_name == 'shufflenet_v2_x1_0'):
        return shufflenet_v2_x1_0(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif (conf.model_name == 'shufflenet_v2_x1_5'):
        return shufflenet_v2_x1_5(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif (conf.model_name == 'shufflenet_v2_x2_0'):
        return shufflenet_v2_x2_0(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes)
    elif(conf.model_name == 'mnasnet0_5'):
        return mnasnet0_5(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes, dropout=conf.drop_rate)
    elif (conf.model_name == 'mnasnet0_75'):
        return mnasnet0_75(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes, dropout=conf.drop_rate)
    elif (conf.model_name == 'mnasnet1_0'):
        return mnasnet1_0(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes, dropout=conf.drop_rate)
    elif (conf.model_name == 'mnasnet1_3'):
        return mnasnet1_3(pretrained=conf.pretrained_model_path_or_url, num_classes=conf.data_classes, dropout=conf.drop_rate)
    else:
        raise NameError('Please enter a valid classifier')
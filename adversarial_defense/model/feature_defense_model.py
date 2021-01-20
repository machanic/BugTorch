import glob
import os
import pretrainedmodels
import torch
from torch import nn
from torchvision import models as torch_models
import cifar_models as models
from adversarial_defense.model.denoise_resnet import DenoiseResNet50, DenoiseResNet101, DenoiseResNet152
from adversarial_defense.model.pcl_resnet import PrototypeConformityLossResNet
from cifar_models_myself import Conv3, DenseNet121, DenseNet169, DenseNet201, GoogLeNet, MobileNet, MobileNetV2, \
    ResNet18, \
    ResNet34, ResNet50, ResNet101, ResNet152, PNASNetA, PNASNetB, EfficientNetB0, DPN26, DPN92, ResNeXt29_2x64d, \
    ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d, SENet18, ShuffleNetG2, ShuffleNetG3, vgg11, vgg13, vgg16, vgg19, \
    PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152, wideresnet28, wideresnet34, \
    wideresnet40, carlinet, wideresnet28drop, wideresnet34drop, wideresnet40drop
from cifar_models_myself.miscellaneous import Identity
from config import pretrained_cifar_model_conf, IN_CHANNELS, IMAGE_SIZE, CLASS_NUM, PY_ROOT
from cifar_models_myself.efficient_densenet import EfficientDenseNet
from cifar_models_myself.ghostnet import ghost_net
from tiny_imagenet_models.densenet import densenet161, densenet121, densenet169, densenet201
from tiny_imagenet_models.resnext import resnext101_32x4d, resnext101_64x4d
import torchvision.models as vision_models
from tiny_imagenet_models.inception import inception_v3
from tiny_imagenet_models.wrn import tiny_imagenet_wrn


class FeatureDefenseModel(nn.Module):
    """
    A StandardModel object wraps a cnn model.
    This model always accept standard image: in [0, 1] range, RGB order, un-normalized, NCHW format
    """
    def __init__(self, dataset, arch, no_grad=True):
        super(FeatureDefenseModel, self).__init__()
        # init cnn model
        self.in_channels = IN_CHANNELS[dataset]
        self.dataset = dataset
        if "denoise" in arch.lower():
            # CIFAR-100@ResNet50_with_denoise_NonLocal_Filter_3.pth.tar
            trained_model_path = "{root}/train_pytorch_model/adversarial_train/feature_denoise/{dataset}@{arch}_NonLocal_Filter_3.pth.tar".format(root=PY_ROOT, dataset=dataset, arch=arch)
            assert os.path.exists(trained_model_path), "{} does not exist!".format(trained_model_path)
        elif dataset.startswith("CIFAR"):
            trained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}-pretrained/{arch}/checkpoint.pth.tar".format(root=PY_ROOT, dataset=dataset, arch=arch)
            assert os.path.exists(trained_model_path), "{} does not exist!".format(trained_model_path)
        elif dataset == "TinyImageNet":
            arch = arch.replace("resnet-", "resnet")
            trained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}@*.pth.tar".format(root=PY_ROOT, dataset=dataset, arch=arch)
            trained_model_path_list = list(glob.glob(trained_model_path))
            assert len(trained_model_path_list)>0, "{} does not exist!".format(trained_model_path)
            trained_model_path = trained_model_path_list[0]
        else:
            trained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}-pretrained/checkpoints/{arch}*.pth".format(
                root=PY_ROOT, dataset=dataset, arch=arch)
            trained_model_path_ls = list(glob.glob(trained_model_path))
            assert trained_model_path_ls,  "{} does not exist!".format(trained_model_path)
            trained_model_path = trained_model_path_ls[0]

        self.cnn = self.make_model(dataset, arch, self.in_channels, CLASS_NUM[dataset], trained_model_path=trained_model_path)
        # init cnn model meta-information
        self.mean = torch.FloatTensor(self.cnn.mean).view(1, self.in_channels, 1, 1).cuda()
        self.mean.requires_grad =True

        self.std = torch.FloatTensor(self.cnn.std).view(1, self.in_channels, 1, 1).cuda()
        self.std.requires_grad = True

        self.input_space = self.cnn.input_space  # 'RGB' or 'GBR'
        self.input_range = self.cnn.input_range  # [0, 1] or [0, 255]
        self.input_size = self.cnn.input_size
        self.no_grad = no_grad
        self.arch = arch

    @staticmethod
    def check_arch(arch, dataset):
        if dataset == "ImageNet":
            return arch in pretrainedmodels.__dict__
        elif dataset == "TinyImageNet":
            trained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}@*.pth.tar".format(
                root=PY_ROOT, dataset=dataset, arch=arch)
            trained_model_path_list = list(glob.glob(trained_model_path))
            return len(trained_model_path_list) > 0
        else:
            trained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}-pretrained/{arch}*".format(
                root=PY_ROOT, dataset=dataset, arch=arch)
            trained_model_path = glob.glob(trained_model_path)
            if len(trained_model_path) > 0:
                return os.path.exists(trained_model_path[0] + "/checkpoint.pth.tar")
            else:
                return False


    def forward(self, x):
        # assign dropout probability
        # if hasattr(self, 'drop'):
        #     self.cnn.drop = self.drop
        # channel order
        if self.input_space == 'BGR':
            x = x[:, [2, 1, 0], :, :]  # pytorch does not support negative stride index (::-1) yet
        # input range
        if max(self.input_range) == 255:
            x = x * 255
        # normalization
        x = (x - self.mean.type(x.dtype).to(x.device)) / self.std.type(x.dtype).to(x.device)
        if self.no_grad:
            with torch.no_grad():
                if "pcl" in self.arch:
                    feats128, feats256, feats1024, x = self.cnn(x)
                else:
                    x = self.cnn(x)
        else:
            if "pcl" in self.arch:
                feats128, feats256, feats1024, x = self.cnn(x)
            else:
                x = self.cnn(x)

        x = x.view(x.size(0), -1)
        if "pcl" in self.arch:
            return feats128, feats256, feats1024, x
        return x

    def load_weight_from_pth_checkpoint(self, model, fname):
        raw_state_dict = torch.load(fname, map_location='cpu')
        if "state_dict" in raw_state_dict:
            raw_state_dict = raw_state_dict["state_dict"]
        state_dict = dict()
        for key, val in raw_state_dict.items():
            new_key = key.replace('module.', '')
            state_dict[new_key] = val
        model.load_state_dict(state_dict)



    def construct_cifar_model(self, arch, dataset, num_classes):
        if "denoise" not in arch.lower():
            conf = pretrained_cifar_model_conf[dataset][arch]
        arch = arch.split("-")[0]
        if arch.startswith('resnext'):
            model = models.__dict__[arch](
                cardinality=conf["cardinality"],
                num_classes=num_classes,
                depth=conf["depth"],
                widen_factor=conf["widen_factor"],
                dropRate=conf["drop"],
            )
        elif arch.startswith('densenet'):
            model = models.__dict__[arch](
                num_classes=num_classes,
                depth=conf["depth"],
                growthRate=conf["growthRate"],
                compressionRate=conf["compressionRate"],
                dropRate=conf["drop"],
            )
        elif arch.startswith('wrn'):
            model = models.__dict__[arch](
                num_classes=num_classes,
                depth=conf["depth"],
                widen_factor=conf["widen_factor"],
                dropRate=conf["drop"],
            )
        elif arch.endswith('resnet') and "pcl_" not in arch and "denoise" not in arch:
            model = models.__dict__[arch](
                num_classes=num_classes,
                depth=conf["depth"],
                block_name=conf["block_name"],
            )
        elif "pcl_resnet" in arch:
            model = PrototypeConformityLossResNet(in_channels=IN_CHANNELS[dataset], depth=conf["depth"], num_classes=CLASS_NUM[dataset])
        elif arch == "DenoiseResNet50":
            model = DenoiseResNet50(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset], whether_denoising=True)
        elif arch == "DenoiseResNet101":
            model = DenoiseResNet101(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset], whether_denoising=True)
        elif arch == "DenoiseResNet152":
            model = DenoiseResNet152(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset], whether_denoising=True)
        else:
            model = models.__dict__[arch](num_classes=num_classes)
        return model

    def make_model(self, dataset, arch, in_channel, num_classes, trained_model_path=None):
        """
        Make model, and load pre-trained weights.
        :param dataset: cifar10 or imagenet
        :param arch: arch name, e.g., alexnet_bn
        :return: model (in cpu and training mode)
        """
        if dataset in ['CIFAR-10',"CIFAR-100", "MNIST","FashionMNIST"]:
            assert trained_model_path is not None and os.path.exists(trained_model_path), "Pretrained weight model file {} does not exist!".format(trained_model_path)
            if arch == 'gdas':
                model = models.gdas(in_channel, num_classes)
                model.mean = [125.3 / 255, 123.0 / 255, 113.9 / 255]
                model.std = [63.0 / 255, 62.1 / 255, 66.7 / 255]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size = [in_channel, IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            elif arch == 'pyramidnet272':
                model = models.pyramidnet272(in_channel, num_classes)
                model.mean = [0.49139968, 0.48215841, 0.44653091]
                model.std = [0.24703223, 0.24348513, 0.26158784]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size = [in_channel, IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            else:
                model = self.construct_cifar_model(arch, dataset, num_classes)  #
                model.mean = [0.4914, 0.4822, 0.4465]
                model.std = [0.2023, 0.1994, 0.2010]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size = [in_channel, IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            # self.load_weight_from_pth_checkpoint(model, trained_model_path)
        elif dataset == "TinyImageNet":
            model = MetaLearnerModelBuilder.construct_tiny_imagenet_model(arch, dataset)
            model.input_space = 'RGB'
            model.input_range = [0, 1]
            model.mean = [0.4914, 0.4822, 0.4465] # if "defense_resnet" not in arch and "denoise" not in arch: [0,0,0] . [1,1,1]
            model.std = [0.2023, 0.1994, 0.2010]
            model.input_size = [in_channel,IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            # model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, location: storage)["state_dict"])
        elif dataset == 'ImageNet':
            os.environ["TORCH_HOME"] = "{}/train_pytorch_model/real_image_model/ImageNet-pretrained".format(PY_ROOT)
            model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained="imagenet")
        return model


class MetaLearnerModelBuilder(object):
    @staticmethod
    def construct_cifar_model(arch, dataset):
        if arch == "conv3":
            network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset], CLASS_NUM[dataset])
        elif arch == "densenet121":
            network = DenseNet121(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "densenet169":
            network = DenseNet169(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "densenet201":
            network = DenseNet201(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "googlenet":
            network = GoogLeNet(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "mobilenet":
            network = MobileNet(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "mobilenet_v2":
            network = MobileNetV2(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "ghost_net":
            network = ghost_net(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnet18":
            network = ResNet18(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnet34":
            network = ResNet34(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnet50":
            network = ResNet50(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnet101":
            network = ResNet101(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnet152":
            network = ResNet152(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "pnasnetA":
            network = PNASNetA(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "pnasnetB":
            network = PNASNetB(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "efficientnet":
            network = EfficientNetB0(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "dpn26":
            network = DPN26(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "dpn92":
            network = DPN92(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnext29_2":
            network = ResNeXt29_2x64d(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnext29_4":
            network = ResNeXt29_4x64d(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnext29_8":
            network = ResNeXt29_8x64d(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnext29_32":
            network = ResNeXt29_32x4d(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "senet18":
            network = SENet18(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "shufflenet_G2":
            network = ShuffleNetG2(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "shufflenet_G3":
            network = ShuffleNetG3(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "vgg11":
            network = vgg11(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "vgg13":
            network = vgg13(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "vgg16":
            network = vgg16(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "vgg19":
            network = vgg19(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "preactresnet18":
            network = PreActResNet18(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "preactresnet34":
            network = PreActResNet34(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "preactresnet50":
            network = PreActResNet50(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "preactresnet101":
            network = PreActResNet101(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "preactresnet152":
            network = PreActResNet152(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "wideresnet28":
            network = wideresnet28(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "wideresnet28drop":
            network = wideresnet28drop(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "wideresnet34":
            network = wideresnet34(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "wideresnet34drop":
            network = wideresnet34drop(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "wideresnet40":
            network = wideresnet40(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "wideresnet40drop":
            network = wideresnet40drop(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "carlinet":
            network = carlinet(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == 'efficient_densenet':
            depth = 40
            block_config = [(depth - 4) // 6 for _ in range(3)]
            network = EfficientDenseNet(IN_CHANNELS[dataset], block_config=block_config,
                                        num_classes=CLASS_NUM[dataset], small_inputs=dataset != "ImageNet", efficient=False)
        return network

    @staticmethod
    def construct_imagenet_model(arch, dataset):
        os.environ["TORCH_HOME"] = "{}/train_pytorch_model/real_image_model/ImageNet-pretrained".format(PY_ROOT)
        if arch == 'efficient_densenet':
            depth = 40
            block_config = [(depth - 4) // 6 for _ in range(3)]
            return EfficientDenseNet(IN_CHANNELS[dataset],block_config=block_config, num_classes=CLASS_NUM[dataset], small_inputs=False, efficient=False)
        elif arch ==  "ghost_net":
            network = ghost_net(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            return network
        model = vision_models.__dict__[arch](pretrained=False)
        return model

    @staticmethod
    def construct_tiny_imagenet_model(arch, dataset):
        if not arch.startswith("densenet") and not arch.startswith("resnext") and arch in torch_models.__dict__:
            network = torch_models.__dict__[arch](pretrained=False)
        num_classes = CLASS_NUM[dataset]
        if arch.startswith("resnet"):
            num_ftrs = network.fc.in_features
            network.fc = nn.Linear(num_ftrs, num_classes)
        elif arch.startswith("densenet"):
            if arch == "densenet161":
                network = densenet161(pretrained=False)
            elif arch == "densenet121":
                network = densenet121(pretrained=False)
            elif arch == "densenet169":
                network = densenet169(pretrained=False)
            elif arch == "densenet201":
                network = densenet201(pretrained=False)
        elif arch == "resnext32_4":
            network = resnext101_32x4d(pretrained=None)
        elif arch == "resnext64_4":
            network = resnext101_64x4d(pretrained=None)
        elif arch == "ghost_net":
            network = ghost_net(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch.startswith("inception"):
            network = inception_v3(pretrained=False)
        elif arch == "WRN-28-10-drop":
            network = tiny_imagenet_wrn(in_channels=IN_CHANNELS[dataset],depth=28,num_classes=CLASS_NUM[dataset],widen_factor=10, dropRate=0.3)
        elif arch == "WRN-40-10-drop":
            network = tiny_imagenet_wrn(in_channels=IN_CHANNELS[dataset], depth=40, num_classes=CLASS_NUM[dataset],
                                        widen_factor=10, dropRate=0.3)
        elif arch.startswith("vgg"):
            network.avgpool = Identity()
            network.classifier[0] = nn.Linear(512 * 2 * 2, 4096)  # 64 /2**5 = 2
            network.classifier[-1] = nn.Linear(4096, num_classes)
        elif "pcl_resnet" in arch:
            network = PrototypeConformityLossResNet(in_channels=IN_CHANNELS[dataset], depth=pretrained_cifar_model_conf[dataset][arch]["depth"], num_classes=CLASS_NUM[dataset])
        elif arch == "DenoiseResNet50":
            network = DenoiseResNet50(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset], whether_denoising=True)
        elif arch == "DenoiseResNet101":
            network = DenoiseResNet101(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset], whether_denoising=True)
        elif arch == "DenoiseResNet152":
            network = DenoiseResNet152(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset], whether_denoising=True)
        return network




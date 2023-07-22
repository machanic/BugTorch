import torch
import torch.nn as nn
import copy


class StandardModel(nn.Module):
    """
    This model always accept standard image: in [0, 1] range, RGB order, un-normalized, NCHW format
    """
    def __init__(self, net):
        super(StandardModel, self).__init__()
        # init victim model
        self.net = net

        # init victim model meta-information
        if len(self.net.input_size) == 3:
            self.mean = torch.FloatTensor(self.net.mean).view(1, -1, 1, 1)
            self.std = torch.FloatTensor(self.net.std).view(1, -1, 1, 1)
        else:
            # must be debug dataset
            assert len(self.net.input_size) == 1
            self.mean = torch.FloatTensor(self.net.mean).view(1, -1)
            self.std = torch.FloatTensor(self.net.std).view(1, -1)
        self.input_space = self.net.input_space  # 'RGB' or 'GBR' or 'GRAY'
        self.input_range = self.net.input_range  # [0, 1] or [0, 255]
        self.input_size = self.net.input_size

    def whiten(self, x):
        # channel order
        if self.input_space == 'BGR':
            x = x[:, [2, 1, 0], :, :]  # pytorch does not support negative stride index (::-1) yet

        # input range
        x = torch.clamp(x, 0, 1)
        if max(self.input_range) == 255:
            x = x * 255

        # normalization
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
        if self.std.device != x.device:
            self.std = self.std.to(x.device)
        x = (x - self.mean) / self.std

        return x

    def forward(self, x):
        raise NotImplementedError



class StandardPolicyModel(StandardModel):
    """
    This model inherits StandardModel class
    """

    def __init__(self, net):
        super(StandardPolicyModel, self).__init__(net=net)
        self.init_state_dict = copy.deepcopy(self.state_dict())
        self.factor = 1.0

        # for policy models, we do whiten in policy.net.forward() instead of policy.forward()
        # since _inv models requires grad w.r.t. input in range [0, 1]
        self.net.whiten_func = self.whiten

    def forward(self, adv_image, image=None, label=None, target=None,
                output_fields=('grad', 'std', 'adv_logit', 'logit')):
        # get distribution mean, (other fields such as adv_logit and logit will also be in output)
        output = self.net(adv_image, image, label, target, output_fields)

        # we have two solutions for scaling:
        # 1. multiply scale factor into mean and keep std unchanged
        # 2. keep mean unchanged and make std divided by scale factor

        # since we only optimize mean (if args.exclude_std is True) and we often use some form of momentum (SGDM/Adam),
        # changing the scale of mean will change the scale of gradient and previous momentum may no longer be suitable
        # so we choose solution 2 for std here: std <-- std / self.factor
        if 'std' in output:
            output['std'] = output['std'] / self.factor

        # only return fields requested, since DistributedDataParallel throw error if unnecessary fields are returned
        return {field_key: output[field_key] for field_key in output_fields if field_key in output}

    def reinit(self):
        self.load_state_dict(self.init_state_dict)
        self.factor = 1.0

    def rescale(self, scale):
        self.factor *= scale

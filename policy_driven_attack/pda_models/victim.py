from torch import nn
import torch

class Standard(nn.Module):
    """
    This model always accept standard image: in [0, 1] range, RGB order, un-normalized, NCHW format
    """
    def __init__(self, net):
        super(Standard, self).__init__()
        # init victim model
        self.net = net
        self.input_space = self.net.input_space  # 'RGB' or 'GBR' or 'GRAY'
        self.input_range = self.net.input_range  # [0, 1] or [0, 255]
        self.input_size = self.net.input_size


    def forward(self, x):
        raise NotImplementedError


class VictimQuery(Standard):
    """
    This model inherits StandardModel class, and maintain a record for query counts and best adversarial examples
    """

    def __init__(self, net):
        super(VictimQuery, self).__init__(net=net)

        # init attack states
        self.image = None
        self.attack_type = None
        self.norm_type = None
        self.label = None
        self.target_label = None
        self.query_count = 0
        self.last_adv_image = None
        self.last_adv_label = None
        self.last_success = None
        self.last_distance = None
        self.best_adv_image = None
        self.best_adv_label = None
        self.best_distance = None
        # self.best_success = None  # best adv image is always and success attack thus could be omitted

    def forward(self, x, no_count=False):
        if not no_count:
            # increase query counter
            self.query_count += x.shape[0]


        # forward
        output = self.net(x)
        return output

    def reset(self, image, label, target_label, attack_type, norm_type):
        self.image = image.clone()
        self.attack_type = attack_type
        self.norm_type = norm_type
        if self.attack_type == 'untargeted':
            assert label is not None
            self.label = label.clone().view([])
        elif self.attack_type == 'targeted':
            assert target_label is not None
            self.target_label = target_label.clone().view([])
        else:
            raise NotImplementedError('Unknown attack type: {}'.format(self.attack_type))

        self.query_count = 0
        self.last_adv_image = None
        self.last_adv_label = None
        self.last_distance = None
        self.last_success = None
        self.best_adv_image = None
        self.best_adv_label = None
        self.best_distance = None

    def calc_mse(self, adv_image):
        assert self.image is not None
        diff = adv_image - self.image
        diff = diff.view(diff.shape[0], -1)
        return (diff ** 2).sum(dim=1) / self.image.numel()

    def calc_distance(self, adv_image):
        assert self.image is not None
        diff = adv_image - self.image
        diff = diff.view(diff.shape[0], -1)
        if self.norm_type == 'l2':
            return torch.sqrt((diff ** 2).sum(dim=1))
        elif self.norm_type == 'linf':
            return diff.abs().max(dim=1)[0]
        else:
            raise NotImplementedError('Unknown norm: {}'.format(self.norm_type))

    def _is_valid_adv_pred(self, pred):
        if self.attack_type == 'untargeted':
            return ~(pred.eq(self.label))
        elif self.attack_type == 'targeted':
            return pred.eq(self.target_label)
        else:
            raise NotImplementedError('Unknown attack type: {}'.format(self.attack_type))

    def query(self, adv_image, sync_best=True, no_count=False):
        adv_image = adv_image.detach()
        assert self.attack_type is not None
        pred = self.forward(adv_image, no_count=no_count).argmax(dim=1)
        distance = self.calc_distance(adv_image)
        success = self._is_valid_adv_pred(pred)

        # check if better adversarial examples are found
        if sync_best:
            if success.any().item():
                if self.best_distance is None or self.best_distance.item() > distance[success].min().item():
                    # if better adversarial examples are found, record it
                    adv_index = distance[success].argmin()
                    best_adv_image = adv_image[success][adv_index].view(self.image.shape)
                    best_adv_label = pred[success][adv_index].view([])
                    best_distance = distance[success][adv_index].view([])
                    failed = False
                    if not self._is_valid_adv_pred(self.forward(best_adv_image, no_count=True).argmax()).item():
                        best_adv_image = self.image + (1 + 1e-6) * (best_adv_image - self.image)
                        best_adv_label = self.forward(best_adv_image, no_count=True).argmax().view([])
                        best_distance = self.calc_distance(best_adv_image).view([])
                        if not self._is_valid_adv_pred(best_adv_label).item():
                            # cannot fix numerical problem after adding a small factor of (adv_image - image)
                            failed = True

                    if (not failed) and \
                            (self.best_distance is None or self.best_distance.item() > best_distance.item()):
                        # if new adv image is still better than previous after fixing, we should record it
                        self.best_adv_image = best_adv_image
                        self.best_adv_label = best_adv_label
                        self.best_distance = best_distance
                    else:
                        # or else we discard it
                        pass

            self.last_adv_image = adv_image.clone()
            self.last_adv_label = pred
            self.last_distance = distance
            self.last_success = success
        return success

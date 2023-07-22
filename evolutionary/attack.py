# -*- coding: gbk -*-
import argparse

import os
import random
import sys


sys.path.append(os.getcwd())
from collections import defaultdict, OrderedDict
import json
from types import SimpleNamespace
import os.path as osp
import torch
import numpy as np
import glog as log
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from config import CLASS_NUM, MODELS_TEST_STANDARD, IN_CHANNELS, IMAGE_DATA_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
from models.standard_model import StandardModel
from models.defensive_model import DefensiveModel
from dataset.target_class_dataset import ImageNetDataset,CIFAR100Dataset,CIFAR10Dataset,TinyImageNetDataset

class Evolutionary(object):
    def __init__(self, model, dataset, clip_min, clip_max, height, width, channels, norm, epsilon,
                 ccov=0.001, decay_weight=0.99, max_queries=10000, mu=0.01, sigma=3e-2, maxlen=30,
                 maximum_queries=10000, batch_size=1):
        """
        :param clip_min: lower bound of the image.
        :param clip_max: upper bound of the image.
        :param norm: choose between [l2, linf].
        :param iterations: number of iterations.
        :param gamma: used to set binary search threshold theta. The binary search
                     threshold theta is gamma / d^{3/2} for l2 attack and gamma / d^2 for linf attack.
        :param max_num_evals: maximum number of evaluations for estimating gradient.
        :param init_num_evals: initial number of evaluations for estimating gradient.
        """
        self.model = model
        self.norm = norm
        self.ord = np.inf if self.norm == "linf" else 2
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.height = height
        self.width = width
        self.channels = channels
        self.shape = (channels, height, width)

        self.model = model
        self.ccov = ccov
        self.decay_weight = decay_weight
        self.max_queries = max_queries
        self.mu = mu
        self.sigma = sigma
        self.maxlen = maxlen
        self.min_value = 0
        self.max_value = 1

        self.maximum_queries = maximum_queries
        self.dataset_name = dataset
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.batch_size = batch_size
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images) # query times
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)

    # We directly randomly select an image from corresponding dataset, and then querying the target model verify it.
    def get_image_of_target_class(self, dataset_name, target_labels, target_model):

        images = []
        for label in target_labels:  # length of target_labels is 1
            if dataset_name == "ImageNet":
                dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name == "CIFAR-10":
                dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name == "CIFAR-100":
                dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name == "TinyImageNet":
                dataset = TinyImageNetDataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            index = np.random.randint(0, len(dataset))
            image, true_label = dataset[index]
            image = image.unsqueeze(0)
            if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
                image = F.interpolate(image,
                                      size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                      align_corners=False)
            with torch.no_grad():
                logits = target_model(image.cuda())
            max_recursive_loop_limit = 100
            loop_count = 0
            while logits.max(1)[1].item() != label.item() and loop_count < max_recursive_loop_limit:
                loop_count += 1
                index = np.random.randint(0, len(dataset))
                image, true_label = dataset[index]
                image = image.unsqueeze(0)
                if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
                    image = F.interpolate(image,
                                          size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                          align_corners=False)
                with torch.no_grad():
                    logits = target_model(image.cuda())

            if loop_count == max_recursive_loop_limit:
                # The program cannot find a valid image from the validation set.
                return None

            assert true_label == label.item()
            images.append(torch.squeeze(image))
        return torch.stack(images)  # B,C,H,W

    def decision_function(self, images, true_labels, target_labels):
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max).cuda()
        logits = self.model(images)
        if target_labels is None:
            return logits.max(1)[1].detach().cpu().item() != true_labels[0].item()
        else:
            return logits.max(1)[1].detach().cpu().item() == target_labels[0].item()

    def initialize(self, sample, target_images, true_labels, target_labels):
        """
        sample: the shape of sample is [C,H,W] without batch-size
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        num_eval = 0
        if target_images is None:
            while True:
                random_noise = torch.from_numpy(np.random.uniform(self.clip_min, self.clip_max, size=self.shape)).float()
                # random_noise = torch.FloatTensor(*self.shape).uniform_(self.clip_min, self.clip_max)
                success = self.decision_function(random_noise[None], true_labels, target_labels)
                num_eval += 1
                if success:
                    break
                if num_eval > 1000:
                    log.info("Initialization failed! Use a misclassified image as `target_image")
                    if target_labels is None:
                        target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
                                                      size=true_labels.size()).long()
                        invalid_target_index = target_labels.eq(true_labels)
                        while invalid_target_index.sum().item() > 0:
                            target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
                                                                size=target_labels[invalid_target_index].size()).long()
                            invalid_target_index = target_labels.eq(true_labels)

                    initialization = self.get_image_of_target_class(self.dataset_name,target_labels, self.model).squeeze()
                    return initialization, 1
                # assert num_eval < 1e4, "Initialization failed! Use a misclassified image as `target_image`"
            # Binary search to minimize l2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample + mid * random_noise
                success = self.decision_function(blended, true_labels, target_labels)
                num_eval += 1
                if success:
                    high = mid
                else:
                    low = mid
            # Sometimes, the found `high` is so tiny that the difference between initialization and sample is very
            # small, this case will cause an infinity loop
            initialization = (1 - high) * sample + high * random_noise
        else:
            initialization = target_images
        return initialization, num_eval

    def _is_adversarial(self, x, y, ytarget):
        output = torch.argmax(self.model(x), dim=1)
        if ytarget is not None:
            return output == ytarget
        else:
            return output != y

    def attack(self, batch_index, images, target_images, true_labels, target_labels):
        query = torch.zeros_like(true_labels).float()
        success_stop_queries = query.clone()  # stop query count once the distortion < epsilon
        batch_image_positions = np.arange(batch_index * self.batch_size,
                                          min((batch_index + 1)*self.batch_size, self.total_images)).tolist()
        batch_size = images.size(0)

        x = images.cuda()
        y = true_labels.cuda()
        if target_labels is not None:
            target_labels = target_labels.cuda()
        pert_shape = (x.size(0), x.size(1), x.size(2), x.size(3))
        m = np.prod(pert_shape)
        k = int(m / 20)
        evolutionary_path = np.zeros(pert_shape)
        decay_weight = self.decay_weight
        diagonal_covariance = np.ones(pert_shape)
        ccov = self.ccov
        # if self._is_adversarial(x, y, target_labels):
        #     return x

        # find an starting point
        # x_adv = self.get_init_noise(x, y, ytarget)
        x_adv, num_eval = self.initialize(images, target_images, true_labels, target_labels)
        # log.info("after initialize")
        query += num_eval
        dist = torch.norm((x_adv - images).view(batch_size, -1), self.ord, 1)
        working_ind = torch.nonzero(dist > self.epsilon).view(-1)  # get locations (i.e., indexes) of non-zero elements of an array.
        success_stop_queries[working_ind] = query[working_ind]  # success times

        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

        x_adv = x_adv.cuda()
        mindist = 1e10
        stats_adversarial = []
        for _ in range(self.max_queries):
            unnormalized_source_direction = x - x_adv
            source_norm = torch.norm(unnormalized_source_direction)
            if mindist > source_norm:
                mindist = source_norm
                best_adv = x_adv

            selection_prob = diagonal_covariance.reshape(-1) / np.sum(diagonal_covariance)
            selection_indices = np.random.choice(m, k, replace=False, p=selection_prob)
            pert = np.random.normal(0.0, 1.0, pert_shape)
            factor = np.zeros([m])
            factor[selection_indices] = True
            pert *= factor.reshape(pert_shape) * np.sqrt(diagonal_covariance)
            pert_large = torch.Tensor(pert).cuda()

            biased = (x_adv + self.mu * unnormalized_source_direction).cuda()
            candidate = biased + self.sigma * source_norm * pert_large / torch.norm(pert_large)
            candidate = x - (x - candidate) / torch.norm(x - candidate) * torch.norm(x - biased)
            candidate = torch.clamp(candidate, self.min_value, self.max_value)

            if self._is_adversarial(candidate, y, target_labels):
                x_adv = candidate
                evolutionary_path = decay_weight * evolutionary_path + np.sqrt(1 - decay_weight ** 2) * pert
                diagonal_covariance = (1 - ccov) * diagonal_covariance + ccov * (evolutionary_path ** 2)
                stats_adversarial.append(1)

                dist = torch.norm((x_adv - x).view(batch_size, -1), self.ord, 1)
                working_ind = torch.nonzero(dist > self.epsilon).view(-1)
                success_stop_queries[working_ind] = query[working_ind]
                for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                    self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                        inside_batch_index].item()
                log.info('{}-th image, {}: distortion {:.4f}, query: {}'.format(batch_index+1, self.norm, dist.item(), int(query[0].item())))
            else:
                stats_adversarial.append(0)
            query += 1
            if len(stats_adversarial) == self.maxlen:
                self.mu *= np.exp(np.mean(stats_adversarial) - 0.2)
                stats_adversarial = []

            if torch.sum(query >= self.maximum_queries).item() == true_labels.size(0):
                break
            # compute new distance.
            # dist = torch.norm((x_adv - x).view(batch_size, -1), self.ord, 1)
            # log.info('{}-th image, {}: distortion {:.4f}, query: {}'.format(batch_index + 1,
            #                                                                 self.norm, dist.item(),
            #                                                                 int(query[0].item())))
            if dist.item() < 1e-4:  # 发现攻击jpeg时候卡住，故意加上这句话
                break
        success_stop_queries = torch.clamp(success_stop_queries, 0, self.maximum_queries)

        return x_adv, query, success_stop_queries, dist, (dist <= self.epsilon)

    def attack_all_images(self, args, arch_name, result_dump_path):
        if args.targeted and args.target_type == "load_random":
            loaded_target_labels = np.load("./target_class_labels/{}/label.npy".format(args.dataset))
            loaded_target_labels = torch.from_numpy(loaded_target_labels).long()
        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                images = F.interpolate(images,
                                       size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            with torch.no_grad():
                logit = self.model(images.cuda())
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels.cuda()).float()  # shape = (batch_size,)
            if correct.int().item() == 0: # we must skip any image that is classified incorrectly before attacking, otherwise this will cause infinity loop in later procedure
                log.info("{}-th original image is classified incorrectly, skip!".format(batch_index+1))
                continue
            selected = torch.arange(batch_index * args.batch_size, min((batch_index + 1) * args.batch_size, self.total_images))
            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                            size=target_labels[invalid_target_index].shape).long()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == "load_random":
                    target_labels = loaded_target_labels[selected]
                    assert target_labels[0].item()!=true_labels[0].item()
                    # log.info("load random label as {}".format(target_labels))
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1).detach().cpu()
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))

                target_images = self.get_image_of_target_class(self.dataset_name, target_labels, self.model)
                if target_images is None:
                    log.info("{}-th image cannot get a valid target class image to initialize!".format(batch_index+1))
                    continue
            else:
                target_labels = None
                target_images = None

            adv_images, query, success_query, distortion_with_max_queries, success_epsilon = self.attack(batch_index, images, target_images, true_labels, target_labels)
            distortion_with_max_queries = distortion_with_max_queries.detach().cpu()
            with torch.no_grad():
                if adv_images.dim() == 3:
                    adv_images = adv_images.unsqueeze(0)
                adv_logit = self.model(adv_images.cuda())
            adv_pred = adv_logit.argmax(dim=1)
            ## Continue query count
            not_done = correct.clone()
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels.cuda()).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels.cuda()).float()  #
            success = (1 - not_done.detach().cpu()) * correct.detach().cpu() * success_epsilon.detach().cpu() * (
                        success_query.detach().cpu() <= self.maximum_queries).float()

            for key in ['query', 'correct', 'not_done',
                        'success', 'success_query', "distortion_with_max_queries"]:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()

        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.bool()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.bool()].mean().item() if self.success_all.sum().item() > 0 else 0,
                          "median_query": self.success_query_all[self.success_all.bool()].median().item() if self.success_all.sum().item() > 0 else 0,
                          "max_query": self.success_query_all[self.success_all.bool()].max().item() if self.success_all.sum().item() > 0 else 0,
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_all":self.success_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_query_all": self.success_query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "distortion": self.distortion_all,
                          "avg_distortion_with_max_queries": self.distortion_with_max_queries_all.mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    if target_type == "load_random":
        target_type = "random"
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'Evolutionary_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        dirname = 'Evolutionary-{}-{}-{}'.format(dataset, norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument('--json-config', type=str, default='./configures/Evolutionary.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, choices=["l2","linf"],required=True)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size must set to 1')
    parser.add_argument('--dataset', type=str, required=True,
               choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"], help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--all-archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target-type',type=str, default='increment', choices=['random', "load_random", 'least_likely',"increment"])
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack-defense',action="store_true")
    parser.add_argument('--defense-model',type=str, default=None)
    parser.add_argument('--defense-norm',type=str,choices=["l2","linf"],default='linf')
    parser.add_argument('--defense-eps',type=str,default="")
    parser.add_argument('--k', type=int, help='the key parameter that influences the results of untargeted and targeted attacks')
    parser.add_argument('--max-queries',type=int, default=10000)

    args = parser.parse_args()
    assert args.batch_size == 1, "Evolutionary Attack only supports mini-batch size equals 1!"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
        args_dict = defaults
    if args.targeted:
        if args.dataset == "ImageNet":
            args.max_queries = 20000
    if args.attack_defense and args.defense_model == "adv_train_on_ImageNet":
        args.max_queries = 20000
    args.exp_dir = osp.join(args.exp_dir,
                            get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type, args))
    os.makedirs(args.exp_dir, exist_ok=True)

    if args.all_archs:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_defense_{}.log'.format(args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run.log')
    elif args.arch is not None:
        if args.attack_defense:
            if args.defense_model == "adv_train_on_ImageNet":
                log_file_path = osp.join(args.exp_dir,
                                         "run_defense_{}_{}_{}_{}.log".format(args.arch, args.defense_model,
                                                                              args.defense_norm,
                                                                              args.defense_eps))
            else:
                log_file_path = osp.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run_{}.log'.format(args.arch))
    set_log_file(log_file_path)
    if args.attack_defense:
        assert args.defense_model is not None

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.all_archs:
        archs = MODELS_TEST_STANDARD[args.dataset]
    else:
        assert args.arch is not None
        archs = [args.arch]
    args.arch = ", ".join(archs)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)
    for arch in archs:
        if args.attack_defense:
            if args.defense_model == "adv_train_on_ImageNet":
                save_result_path = args.exp_dir + "/{}_{}_{}_{}_result.json".format(arch, args.defense_model,
                                                                                    args.defense_norm,args.defense_eps)
            else:
                save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model,
                                   norm=args.defense_norm, eps=args.defense_eps)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        attacker = Evolutionary(model, args.dataset, 0, 1.0, model.input_size[-2], model.input_size[-1], IN_CHANNELS[args.dataset],
                                args.norm, args.epsilon, maximum_queries=args.max_queries, batch_size=args.batch_size)
        attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()

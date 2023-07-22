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
from torch.nn import functional as F
from config import CLASS_NUM, MODELS_TEST_STANDARD, IMAGE_DATA_ROOT, IN_CHANNELS
from dataset.dataset_loader_maker import DataLoaderMaker
from models.standard_model import StandardModel
from models.defensive_model import DefensiveModel
from dataset.target_class_dataset import ImageNetDataset,CIFAR100Dataset,CIFAR10Dataset,TinyImageNetDataset

import math
import dct as torch_dct

def atleast_kdim(x, ndim):
    shape = x.shape + (1,) * (ndim - len(x.shape))
    return x.reshape(shape)

class SurFree(object):
    def __init__(self, model, dataset, clip_min, clip_max, height, width, channels, batch_size, epsilon, norm,
                maximum_queries=10000, BS_gamma: float = 0.01, quantification=True, theta_max: float = 30,
                BS_max_iteration: int = 7, steps: int = 100, n_ortho: int = 100, clip=True,
                rho: float = 0.95, T: int = 1, with_alpha_line_search: bool = True, with_distance_line_search: bool = False,
                with_interpolation: bool = False, final_line_search: bool=True,):
        self.model = model
        self.batch_size = batch_size
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.images = torch.from_numpy(self.dataset_loader.dataset.images).cuda()
        self.labels = torch.from_numpy(self.dataset_loader.dataset.labels).cuda()
        self.total_images = len(self.dataset_loader.dataset)
        self.norm = norm
        self.ord = np.inf if self.norm == "linf" else 2
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.shape = (channels, height, width)

        # Attack Parameters
        self._steps = steps
        self._BS_gamma = BS_gamma
        self._theta_max = theta_max
        # self._max_queries = max_queries
        self._BS_max_iteration = BS_max_iteration
        self._steps = steps
        self.clip = clip
        self.T = T
        self.rho = rho

        self.with_alpha_line_search = with_alpha_line_search
        self.with_distance_line_search = with_distance_line_search
        self.with_interpolation = with_interpolation
        self.final_line_search = final_line_search
        self.quantification = quantification
        self._alpha_history = {}
        self.n_ortho = n_ortho

        self.maximum_queries = maximum_queries
        self.dataset_name = dataset
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.batch_size = batch_size
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)

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
            return logits.max(1)[1].detach().cpu() != true_labels
        else:
            return logits.max(1)[1].detach().cpu() == target_labels

    def initialize(self, sample, target_images, true_labels, target_labels):
        """
        sample: the shape of sample is [C,H,W] without batch-size
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        sample = sample.unsqueeze(0)
        num_eval = 0
        if target_images is None:
            while True:
                random_noise = torch.from_numpy(np.random.uniform(self.clip_min, self.clip_max, size=self.shape)).float()
                # random_noise = torch.FloatTensor(*self.shape).uniform_(self.clip_min, self.clip_max)
                success = self.decision_function(random_noise[None], true_labels, target_labels)[0].item()
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

                    initialization = self.get_image_of_target_class(self.dataset_name, [target_labels], self.model).squeeze()
                    return initialization, 1
                # assert num_eval < 1e4, "Initialization failed! Use a misclassified image as `target_image`"
            # Binary search to minimize l2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample + mid * random_noise
                success = self.decision_function(blended, true_labels, target_labels)[0].item()
                num_eval += 1
                if success:
                    high = mid
                else:
                    low = mid
            # Sometimes, the found `high` is so tiny that the difference between initialization and sample is very very small, this case will cause inifinity loop
            initialization = (1 - high) * sample + high * random_noise
        else:
            initialization = target_images
        return initialization, num_eval

    def batch_initialize(self, samples, target_images, true_labels, target_labels):
        initialization = samples.clone()
        num_evals = torch.zeros_like(true_labels).float()

        with torch.no_grad():
            logit = self.model(samples.cuda())
        pred = logit.argmax(dim=1)
        correct = pred.eq(true_labels.cuda()).float()
        for i in range(len(correct)):
            if correct[i]:
                if target_images is None:
                    initialization[i], num_evals[i] = self.initialize(samples[i], None, true_labels[i], None)
                else:
                    initialization[i], num_evals[i] = self.initialize(samples[i], target_images[i], true_labels[i], target_labels[i])

        return initialization, num_evals

    def distance(self, a, b):
        return (a - b).flatten(1).norm(dim=1)

    def _quantify(self, x):
        return (x * 255).round() / 255

    def get_nqueries(self):
        return {i: n for i, n in enumerate(self._nqueries)}

    def _is_adversarial(self, perturbed):
        # Faster than true_is_adversarial in batch  (time gain 20%)
        # Count the queries made for each image
        # Count if the vector is different from the null vector
        if self.quantification:
            perturbed = self._quantify(perturbed)
        if self.target_labels is not None:
            is_advs = self.model(perturbed.cuda()).argmax(1).detach().cpu() == self.target_labels
        else:
            is_advs = self.model(perturbed.cuda()).argmax(1).detach().cpu() != self.label

        indexes = []
        for i, p in enumerate(perturbed):
            if not (p == 0).all() and not self._images_finished[i]:
                self._nqueries[i] += 1
                indexes.append(i)

        self._images_finished = self._nqueries > self.maximum_queries
        return is_advs

    def _get_candidates(self):
        """
        Find the lowest torchsilon to misclassified x following the direction: q of class 1 / q + torchs*direction of class 0
        """
        epsilon = torch.zeros(len(self.X)).to(self.X.device)
        direction_2 = torch.zeros_like(self.X)
        distances = (self.X - self.best_advs).flatten(1).norm(dim=1)
        while (epsilon == 0).any():
            epsilon = torch.where(self._images_finished, torch.ones_like(epsilon), epsilon)

            new_directions = self._basis.get_vector(self._directions_ortho,
                                                    indexes=[i for i, eps in enumerate(epsilon) if eps == 0])

            direction_2 = torch.where(
                atleast_kdim(epsilon == 0, len(direction_2.shape)),
                new_directions,
                direction_2
            )
            for i, eps_i in enumerate(epsilon):
                if i not in self._alpha_history:
                    self._alpha_history[i] = []
                if eps_i == 0:
                    # Concatenate the first directions and the last directions generated
                    self._directions_ortho[i] = torch.cat((
                        self._directions_ortho[i][:1],
                        self._directions_ortho[i][1 + len(self._directions_ortho[i]) - self.n_ortho:],
                        direction_2[i].unsqueeze(0)), dim=0)

                    self._alpha_history[i].append([
                        len(self._history[i]),
                        float(self.theta_max[i].cpu()),
                        float(distances[i].cpu())
                    ])

            function_evolution = self._get_evolution_function(direction_2)

            new_epsilons = self._get_best_theta(function_evolution, epsilon == 0)

            for i, eps_i in enumerate(epsilon):
                if eps_i == 0:
                    if new_epsilons[i] == 0:
                        self._alpha_history[i][-1] += [False]
                    else:
                        self._alpha_history[i][-1] += [True]

            self.theta_max = torch.where(new_epsilons == 0, self.theta_max * self.rho, self.theta_max)
            self.theta_max = torch.where((new_epsilons != 0) * (epsilon == 0), self.theta_max / self.rho,
                                         self.theta_max)
            epsilon = torch.where((new_epsilons != 0) * (epsilon == 0), new_epsilons, epsilon)

        function_evolution = self._get_evolution_function(direction_2)
        if self.with_alpha_line_search:
            epsilon = self._alpha_binary_search(function_evolution, epsilon)

        epsilon = epsilon.unsqueeze(0)
        if self.with_interpolation:
            epsilon = torch.cat((epsilon, epsilon / 2), dim=0)

        candidates = torch.cat([function_evolution(eps).unsqueeze(0) for eps in epsilon], dim=0)

        if self.with_interpolation:
            d = (self.best_advs - self.X).flatten(1).norm(dim=1)
            delta = (self._binary_search(candidates[1], boost=True) - self.X).flatten(1).norm(dim=1)
            theta_star = epsilon[0]

            num = theta_star * (4 * delta - d * (torch.cos(theta_star.raw) + 3))
            den = 4 * (2 * delta - d * (torch.cos(theta_star.raw) + 1))

            theta_hat = num / den
            q_interp = function_evolution(theta_hat)
            candidates = torch.cat((candidates, q_interp.unsqueeze(0)), dim=0)

        if self.with_distance_line_search:
            for i, candidate in enumerate(candidates):
                candidates[i] = self._binary_search(candidate, boost=True)
        return candidates

    def _get_evolution_function(self, direction_2):
        distances = (self.best_advs - self.X).flatten(1).norm(dim=1, keepdim=True)
        direction_1 = (self.best_advs - self.X).flatten(1) / distances
        direction_1 = direction_1.reshape(self.X.shape)

        if self.clip:
            return lambda theta: (
                        self.X + self._add_step_in_circular_direction(direction_1, direction_2, distances, theta)).clip(
                0, 1)
        else:
            return lambda theta: (
                        self.X + self._add_step_in_circular_direction(direction_1, direction_2, distances, theta))

    def _add_step_in_circular_direction(self, direction1: torch.Tensor, direction2: torch.Tensor, r: torch.Tensor,
                                        degree: torch.Tensor) -> torch.Tensor:
        degree = atleast_kdim(degree, len(direction1.shape))
        r = atleast_kdim(r, len(direction1.shape))
        results = torch.cos(degree * np.pi / 180) * direction1 + torch.sin(degree * np.pi / 180) * direction2
        results = results * r * torch.cos(degree * np.pi / 180)
        return results

    def _get_best_theta(self, function_evolution, mask):
        coefficients = torch.zeros(2 * self.T).to(self.X.device)
        for i in range(0, self.T):
            coefficients[2 * i] = 1 - (i / self.T)
            coefficients[2 * i + 1] = - coefficients[2 * i]

        best_params = torch.zeros_like(self.theta_max)
        for i, coeff in enumerate(coefficients):

            params = coeff * self.theta_max
            x_evol = function_evolution(params)
            x = torch.where(
                atleast_kdim((best_params == 0) * mask, len(self.X.shape)),
                x_evol,
                torch.zeros_like(self.X))

            is_advs = self._is_adversarial(x)
            best_params = torch.where(
                (best_params == 0) * mask * is_advs,
                params,
                best_params
            )
            if (best_params != 0).all():
                break

        return best_params

    def _alpha_binary_search(self, function_evolution, lower):
        # Upper --> not adversarial /  Lower --> adversarial
        mask = self._images_finished.logical_not()

        def get_alpha(theta: torch.Tensor) -> torch.Tensor:
            return 1 - torch.cos(theta * np.pi / 180)

        lower = torch.where(mask, lower, torch.zeros_like(lower))
        check_opposite = lower > 0  # if param < 0: abs(param) doesn't work

        # Get the upper range
        upper = torch.where(
            torch.logical_and(abs(lower) != self.theta_max, mask),
            lower + torch.sign(lower) * self.theta_max / self.T,
            torch.zeros_like(lower)
        )

        mask_upper = torch.logical_and(upper == 0, mask)
        max_angle = torch.ones_like(lower) * 180
        while mask_upper.any():
            # Find the correct lower/upper range
            # if True in mask_upper, the range haven't been found
            new_upper = lower + torch.sign(lower) * self.theta_max / self.T
            new_upper = torch.where(new_upper < max_angle, new_upper, max_angle)
            new_upper_x = function_evolution(new_upper)
            x = torch.where(
                atleast_kdim(mask_upper, len(self.X.shape)),
                new_upper_x,
                torch.zeros_like(self.X)
            )

            is_advs = self._is_adversarial(x)
            lower = torch.where(torch.logical_and(mask_upper, is_advs), new_upper, lower)
            upper = torch.where(torch.logical_and(mask_upper, is_advs.logical_not()), new_upper, upper)
            mask_upper = mask_upper * is_advs * torch.logical_not(self._images_finished)

        lower = torch.where(self._images_finished, torch.zeros_like(lower), lower)
        upper = torch.where(self._images_finished, torch.zeros_like(upper), upper)

        step = 0
        over_gamma = abs(get_alpha(upper) - get_alpha(lower)) > self._BS_gamma
        while step < self._BS_max_iteration and over_gamma.any():
            mid_bound = (upper + lower) / 2
            mid = torch.where(
                atleast_kdim(torch.logical_and(mid_bound != 0, over_gamma), len(self.X.shape)),
                function_evolution(mid_bound),
                torch.zeros_like(self.X)
            )
            is_adv = self._is_adversarial(mid)

            mid_opp = torch.where(
                atleast_kdim(torch.logical_and(check_opposite, over_gamma), len(mid.shape)),
                function_evolution(-mid_bound),
                torch.zeros_like(mid)
            )
            is_adv_opp = self._is_adversarial(mid_opp)

            lower = torch.where(mask * over_gamma * is_adv, mid_bound, lower)
            lower = torch.where(mask * over_gamma * is_adv.logical_not() * check_opposite * is_adv_opp, -mid_bound,
                                lower)
            upper = torch.where(mask * over_gamma * is_adv.logical_not() * check_opposite * is_adv_opp, - upper, upper)
            upper = torch.where(mask * over_gamma * (abs(lower) != abs(mid_bound)), mid_bound, upper)

            check_opposite = mask * over_gamma * check_opposite * is_adv_opp * (lower > 0)
            over_gamma = abs(get_alpha(upper) - get_alpha(lower)) > self._BS_gamma

            step += 1

        return lower

    def _binary_search(self, perturbed: torch.Tensor, boost=False) -> torch.Tensor:
        # Choose upper thresholds in binary search based on constraint.
        highs = torch.ones(len(perturbed)).to(perturbed.device)
        d = np.prod(perturbed.shape[1:])
        thresholds = self._BS_gamma / (d * math.sqrt(d))
        lows = torch.zeros_like(highs)
        mask = atleast_kdim(self._images_finished, len(perturbed.shape))

        # Boost Binary search
        if boost:
            boost_vec = torch.where(mask, torch.zeros_like(perturbed), 0.2 * self.X + 0.8 * perturbed)
            is_advs = self._is_adversarial(boost_vec)
            is_advs = atleast_kdim(is_advs, len(self.X.shape))
            originals = torch.where(is_advs.logical_not(), boost_vec, self.X)
            perturbed = torch.where(is_advs, boost_vec, perturbed)
        else:
            originals = self.X

        # use this variable to check when mids stays constant and the BS has converged
        iteration = 0
        while torch.any(highs - lows > thresholds) and iteration < self._BS_max_iteration:
            iteration += 1
            mids = (lows + highs) / 2
            epsilon = atleast_kdim(mids, len(originals.shape))

            mids_perturbed = torch.where(
                mask,
                torch.zeros_like(perturbed),
                (1.0 - epsilon) * originals + epsilon * perturbed)

            is_adversarial_ = self._is_adversarial(mids_perturbed)

            highs = torch.where(is_adversarial_, mids, highs)
            lows = torch.where(is_adversarial_, lows, mids)

        epsilon = atleast_kdim(highs, len(originals.shape))
        return torch.where(mask, perturbed, (1.0 - epsilon) * originals + epsilon * perturbed)

    def attack(self, batch_index, images, target_images, true_labels, target_labels, **kwargs):
        self._nqueries = torch.zeros(len(images)).to(images.device)
        self._history  = {i: [] for i in range(len(images))}
        self.theta_max = torch.ones(len(images)).to(images.device) * self._theta_max

        self.label = true_labels
        self.X = images
        self.target_labels = target_labels
        batch_size = images.size(0)

        query = torch.zeros_like(true_labels).float()
        success_stop_queries = query.clone()  # stop query count once the distortion < epsilon
        batch_image_positions = np.arange(batch_index * self.batch_size,
                                          min((batch_index + 1) * self.batch_size, self.total_images)).tolist()

        # self.best_advs = torch.where(atleast_kdim(self._images_finished, len(X.shape)), X, self.best_advs)
        # self.best_advs = self._binary_search(self.best_advs, boost=True)
        self.best_advs, num_evals = self.batch_initialize(self.X, target_images, true_labels, target_labels)
        # self.best_advs = get_init_with_noise(model, X, labels) if starting_points is None else starting_points
        query += num_evals
        self._nqueries += num_evals
        dist = torch.norm((self.best_advs - images).view(batch_size, -1), self.ord, 1)
        working_ind = torch.nonzero(dist > self.epsilon).view(-1)
        success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                inside_batch_index].item()

        # Check if X are already adversarials.
        if target_labels is None:
            self._images_finished = model(images.cuda()).argmax(1).detach().cpu() != true_labels
        else:
            self._images_finished = model(images.cuda()).argmax(1).detach().cpu() == target_labels

        print("Already advs: ", self._images_finished.cpu().tolist())
        self.best_advs = torch.where(atleast_kdim(self._images_finished, len(images.shape)), images, self.best_advs)
        # self.best_advs = self._binary_search(self.best_advs, boost=True)

        # Initialize the direction orthogonalized with the first direction
        fd = self.best_advs - self.X
        self._directions_ortho = {i: v.unsqueeze(0) / v.norm() for i, v in enumerate(fd)}

        # Load Basis
        self._basis = Basis(self.X, **kwargs["basis_params"]) if "basis_params" in kwargs else Basis(self.X)

        while min(query) < self.maximum_queries:
            # Get candidates. Shape: (n_candidates, batch_size, image_size)
            candidates = self._get_candidates()
            candidates = candidates.transpose(1, 0)

            best_candidates = torch.zeros_like(self.best_advs)
            for i, o in enumerate(self.X):
                o_repeated = torch.cat([o.unsqueeze(0)] * len(candidates[i]), dim=0)
                index = self.distance(o_repeated, candidates[i]).argmax()
                best_candidates[i] = candidates[i][index]

            is_success = self.distance(best_candidates, self.X) < self.distance(self.best_advs, self.X)
            self.best_advs = torch.where(
                atleast_kdim(is_success * self._images_finished.logical_not(), len(best_candidates.shape)),
                best_candidates,
                self.best_advs
            )

            query = self._nqueries
            dist = torch.norm((self.best_advs - images).view(batch_size, -1), self.ord, 1)
            working_ind = torch.nonzero(dist > self.epsilon).view(-1)
            success_stop_queries[working_ind] = query[working_ind]
            for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                    inside_batch_index].item()

            log.info('Attacking image {} - {} / {}, query {}, distortion {}'.format(
                batch_index * args.batch_size, (batch_index + 1) * args.batch_size, self.total_images, query, dist))

            # if self._images_finished.all():
            #     print("Max queries attained for all the images.")
            #     break

            # log.info('{}-th image, {}: distortion {:.4f}, query: {}'.format(batch_index+1, self.norm, dist.item(), int(query[0].item())))
            # if dist.item() < 1e-4:  # 发现攻击jpeg时候卡住，故意加上这句话
            #     break
        if self.final_line_search:
            self.best_advs = self._binary_search(self.best_advs,  boost=True)

        #print("Final adversarial", self._criterion_is_adversarial(self.best_advs).raw.cpu().tolist())
        if self.quantification:
            self.best_advs = self._quantify(self.best_advs)

        success_stop_queries = torch.clamp(success_stop_queries, 0, self.maximum_queries)
        return self.best_advs, query, success_stop_queries, dist, (dist <= self.epsilon)

    def attack_all_images(self, args, arch_name, result_dump_path, config):
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
            # if correct.int().item() == 0:  # we must skip any image that is classified incorrectly before attacking, otherwise this will cause infinity loop in later procedure
            #     log.info("{}-th original image is classified incorrectly, skip!".format(batch_index + 1))
            #     continue
            selected = torch.arange(batch_index * args.batch_size,
                                    min((batch_index + 1) * args.batch_size, self.total_images))
            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                            size=target_labels[
                                                                                invalid_target_index].shape).long()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == "load_random":
                    target_labels = loaded_target_labels[selected]
                    assert target_labels[0].item() != true_labels[0].item()
                    # log.info("load random label as {}".format(target_labels))
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1).detach().cpu()
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))

                target_images = self.get_image_of_target_class(self.dataset_name, target_labels, self.model)
                if target_images is None:
                    log.info("{}-th image cannot get a valid target class image to initialize!".format(batch_index + 1))
                    continue
            else:
                target_labels = None
                target_images = None

            adv_images, query, success_query, distortion_with_max_queries, success_epsilon = self.attack(batch_index, images, target_images, true_labels, target_labels, **config["run"])

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
                          "success_all": self.success_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_query_all": self.success_query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "distortion": self.distortion_all,
                          "avg_distortion_with_max_queries": self.distortion_with_max_queries_all.mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


class Basis:
    def __init__(self, originals: torch.Tensor, random_noise: str = "normal", basis_type: str = "dct", **kwargs):
        """
        Args:
            random_noise (str, optional): When basis is created, a noise will be added.This noise can be normal or
                                          uniform. Defaults to "normal".
            basis_type (str, optional): Type of the basis: DCT, Random, Genetic,. Defaults to "random".
            device (int, optional): [description]. Defaults to -1.
            args, kwargs: In args and kwargs, there is the basis params:
                    * Random: No parameters
                    * DCT:
                            * function (tanh / constant / linear): function applied on the dct
                            * beta
                            * gamma
                            * frequence_range: tuple of 2 float
                            * dct_type: 8x8 or full
        """
        self.X = originals
        self._f_dct2 = lambda a: torch_dct.dct_2d(a)
        self._f_idct2 = lambda a: torch_dct.idct_2d(a)

        self.basis_type = basis_type
        self._function_generation = getattr(self, "_get_vector_" + self.basis_type)
        self._load_params(**kwargs)

        assert random_noise in ["normal", "uniform"]
        self.random_noise = random_noise

    def get_vector(self, ortho_with=None, indexes=None) -> torch.Tensor:
        random.seed()
        if indexes is None:
            indexes = range(len(self.X))
        if ortho_with is None:
            ortho_with = {i: None for i in indexes}

        r: torch.Tensor = self._function_generation(indexes)
        vectors = [
            self._gram_schmidt(r[i], ortho_with[i]) if ortho_with[i] is not None else r[i]
            for i in range(len(self.X))
        ]
        vectors = torch.cat([v.unsqueeze(0) for v in vectors], dim=0)
        norms = vectors.flatten(1).norm(dim=1)
        vectors /= atleast_kdim(norms, len(vectors.shape))
        return vectors

    def _gram_schmidt(self, v: torch.Tensor, ortho_with: torch.Tensor):
        v_repeated = torch.cat([v.unsqueeze(0)] * len(ortho_with), axis=0)

        # inner product
        gs_coeff = (ortho_with * v_repeated).flatten(1).sum(1)
        proj = atleast_kdim(gs_coeff, len(ortho_with.shape)) * ortho_with
        v = v - proj.sum(0)
        return v

    def _get_vector_dct(self, indexes) -> torch.Tensor:
        probs = self.X[indexes].uniform_(0, 3).long() - 1
        r_np = self.dcts[indexes] * probs
        r_np = self._inverse_dct(r_np)
        new_v = torch.zeros_like(self.X)
        new_v[indexes] = (r_np + self.X[indexes].normal_(std=self._beta))
        return new_v

    def _get_vector_random(self, indexes) -> torch.Tensor:
        r = torch.zeros_like(self.X)
        r = getattr(r, self.random_noise + "_")(0, 1)
        new_v = torch.zeros_like(self.X)
        new_v[indexes] = r[indexes]
        return new_v

    def _load_params(
            self,
            beta: float = 0.001,
            frequence_range=(0, 0.5),
            dct_type: str = "full",
            function: str = "tanh",
            tanh_gamma: float = 1
    ) -> None:
        if not hasattr(self, "_get_vector_" + self.basis_type):
            raise ValueError("Basis {} doesn't exist.".format(self.basis_type))

        if self.basis_type == "dct":
            self._beta = beta
            if dct_type == "8x8":
                mask_size = (8, 8)
                dct_function = self.dct2_8_8
                self._inverse_dct = self.idct2_8_8
            elif dct_type == "full":
                mask_size = self.X.shape[-2:]
                dct_function = lambda x, mask: self._f_dct2(x) * mask
                self._inverse_dct = self._f_idct2
            else:
                raise ValueError("DCT {} doesn't exist.".format(dct_type))

            dct_mask = self.get_zig_zag_mask(frequence_range, mask_size).to(self.X.device)
            self.dcts = dct_function(self.X, dct_mask)

            def get_function(function: str):
                if function == "tanh":
                    return lambda x: torch.tanh(tanh_gamma * x)
                elif function == "identity":
                    return lambda x: x
                elif function == "constant":
                    return lambda x: (abs(x) > 0).long()
                else:
                    raise ValueError("Function given for DCT is incorrect.")

            self.dcts = get_function(function)(self.dcts)

    def get_zig_zag_mask(self, frequence_range, mask_shape=(8, 8)):
        total_component = mask_shape[0] * mask_shape[1]
        n_coeff_kept = int(total_component * min(1, frequence_range[1]))
        n_coeff_to_start = int(total_component * max(0, frequence_range[0]))

        imsize = self.X.shape
        mask_shape = (imsize[0], imsize[1], mask_shape[0], mask_shape[1])
        mask = torch.zeros(mask_shape)
        s = 0

        while n_coeff_kept > 0:
            for i in range(min(s + 1, mask_shape[2])):
                for j in range(min(s + 1, mask_shape[3])):
                    if i + j == s:
                        if n_coeff_to_start > 0:
                            n_coeff_to_start -= 1
                            continue

                        if s % 2:
                            mask[:, :, i, j] = 1
                        else:
                            mask[:, :, j, i] = 1
                        n_coeff_kept -= 1
                        if n_coeff_kept == 0:
                            return mask
            s += 1
        return mask

    def dct2_8_8(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert mask.shape[-2:] == (8, 8)

        imsize = image.shape
        dct = torch.zeros_like(image)
        for i in np.r_[:imsize[2]:8]:
            for j in np.r_[:imsize[3]:8]:
                dct_i_j = self._f_dct2(image[:, :, i:(i + 8), j:(j + 8)])
                dct[:, :, i:(i + 8), j:(j + 8)] = dct_i_j * mask  # [:dct_i_j.shape[0], :dct_i_j.shape[1]]
        return dct

    def idct2_8_8(self, dct: torch.Tensor) -> torch.Tensor:
        im_dct = torch.zeros_like(dct)
        for i in np.r_[:dct.shape[2]:8]:
            for j in np.r_[:dct.shape[3]:8]:
                im_dct[:, :, i:(i + 8), j:(j + 8)] = self._f_idct2(dct[:, :, i:(i + 8), j:(j + 8)])
        return im_dct

def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    if target_type == "load_random":
        target_type = "random"
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'SurFree_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
    else:
        dirname = 'SurFree-{}-{}-{}'.format(dataset, norm, target_str)
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
    parser.add_argument('--json-config', type=str, default='../configures/SurFree.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, choices=["l2","linf"],required=True)
    parser.add_argument('--batch-size', type=int, default=100)
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
    parser.add_argument('--max-queries',type=int, default=10000)

    args = parser.parse_args()
    # assert args.batch_size == 1, "Triangle Attack only supports mini-batch size equals 1!"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
        config = {"init": {}, "run": {"epsilons": None}}
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        config = json.load(open(args.json_config, "r"))
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
                            get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
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

    # if args.json_config is not None:
    #     if not os.path.exists(args.config_path):
    #         raise ValueError("{} doesn't exist.".format(args.config_path))
    #     config = json.load(open(args.config_path, "r"))
    # else:
    #     config = {"init": {}, "run": {"epsilons": None}}

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
        args.side_length = model.input_size[-1]
        attacker = SurFree(model, args.dataset, 0, 1.0, model.input_size[-2], model.input_size[-1],
                           IN_CHANNELS[args.dataset], args.batch_size, args.epsilon, args.norm,
                           maximum_queries=args.max_queries)
        attacker.attack_all_images(args, arch, save_result_path, config)
        model.cpu()

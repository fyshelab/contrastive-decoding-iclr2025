import math
import time
from typing import Dict, Any, Tuple, Sequence, Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import h5py

from research.metrics.metrics import (
    r2_score,
    pearsonr,
    cosine_distance,
    mean_squared_distance,
    contrastive_score,
    two_versus_two,
    evaluate_decoding,
    top_knn_test,
)
from pipeline.utils import merge_dicts, nested_insert, get_data_iterator, nested_select
from pipeline.utils import TorchTimer


class NSDExperiment:
    def __init__(
            self,
            mode: str,
            train_dataset: Dataset,
            val_dataset: Dataset,
            device: torch.device,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler,
            batch_size: int,
            evaluation_interval: int,
            evaluation_subset_size: Optional[int] = None,
            evaluation_subset_seed: int = 0,
            evaluation_objective: Optional[str] = None,
            distance_metric: str = 'cosine',
            drop_last: bool = False,
            augmentation: bool = False,
            augmentation_noise_scale: float = 0.1,
            sdc_criterion: Optional[nn.Module] = None,
            sdc_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        assert mode in ('decode', 'encode')
        self.mode = mode
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.batch_size = batch_size
        self.evaluation_interval = evaluation_interval
        self.evaluation_subset_size = evaluation_subset_size
        self.evaluation_subset_seed = evaluation_subset_seed
        self.evaluation_objective = evaluation_objective
        self.distance_metric = distance_metric
        if self.evaluation_objective is None:
            self.evaluation_objective = ('top_knn_accuracy', self.distance_metric, str(5), 'val')
            #if mode == 'decode':
            #    self.evaluation_objective = ('top_knn_accuracy', self.distance_metric, str(5), 'val')
            #if mode == 'encode':
            #    self.evaluation_objective = ('r2_score', 'val')

        self.augmentation = augmentation
        self.augmentation_noise_scale = augmentation_noise_scale
        self.sdc_criterion = sdc_criterion
        self.sdc_optimizer = sdc_optimizer

        dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, drop_last=drop_last)

        self.data_iterator = get_data_iterator(dataloader, new_epoch_callback=self.new_epoch)
        self.iteration = 0
        self.best_score = -math.inf
        self.best_state_dict = model.state_dict()
        self.best_iteration = 0
        self.epoch = 0
        self.timer = TorchTimer(device=device)

        self.wandb_run = None

    def get_data(self, batch):
        stimulus = batch['stimulus']['data']
        betas = batch['betas'][0]
        if self.mode == 'decode':
            x = betas
            y = stimulus
        else:
            x = stimulus
            y = betas
        return x, y

    def new_epoch(self):
        self.epoch += 1
        self.scheduler.step()

    def train_model(self, max_iterations: int, logger: Optional[Callable] = None):
        for i in tqdm(range(max_iterations)):
            self.timer.start()

            evaluation_dict = {}
            if self.iteration % self.evaluation_interval == 0:
                evaluation_dict = self.evaluate(subset_size=self.evaluation_subset_size,
                                                subset_seed=self.evaluation_subset_seed)
                score = nested_select(evaluation_dict, self.evaluation_objective)
                if score > self.best_score:
                    self.best_score = score
                    self.best_state_dict = self.model.state_dict()
                    self.best_iteration = self.iteration
                self.timer.stamp("evaluation")

            batch = next(self.data_iterator)
            x, y = self.get_data(batch)
            x = x.to(self.device)
            y = y.to(self.device)
            self.timer.stamp("load_data")

            self.model.train()
            if self.augmentation:
                x = x + torch.randn_like(x) * self.augmentation_noise_scale

            if isinstance(self.model, SDCDecoder):

                w, w_pred = self.model(x, y)
                self.timer.stamp("forward")
                loss = self.criterion(w, w_pred) # self.sdc_criterion(w, w_pred)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                #for name, param in self.model.named_parameters():
                #    if name.startswith('clip_to_sdc') or name.startswith('hidden_to_sdc'):
                #        param.requires_grad = True
                #    else:
                #        param.requires_grad = False

                #for param in self.model.parameters():
                #    param.requries_grad = True
                
                loss_dict = {'loss': loss}
            else:
                y_pred = self.model(x)
                self.timer.stamp("forward")
                y_pred = y_pred.reshape(y.shape)
                loss = self.criterion(y, y_pred)
                self.timer.stamp("criterion")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.timer.stamp("backward")

                if isinstance(loss, Tuple):
                    loss, loss_dict = loss
                else:
                    loss_dict = {'loss': loss}
                self.model.eval()
            #model_out = self.model(x)
            #if isinstance(model_out, Tuple):
            #    y_pred, mu, log_var = model_out
            #    loss = self.criterion(y, y_pred, mu, log_var)
            #else:

            log_dict = {
                **loss_dict,
                **evaluation_dict,
                'lr': self.scheduler.get_last_lr()[0],
                'batch_size': x.shape[0],
                'epoch': self.epoch,
                'timer': self.timer.timestamps,
            }
            if logger:
                logger(log_dict)

            self.iteration += 1

    def run_all(self, dataset: Dataset):
        Y_pred = []
        Y = []
        W = []
        W_pred = []
        stimulus_ids = []
        for elem in dataset:
            x, y = self.get_data(elem)
            Y.append(y.cpu())
            stimulus_id = elem['stimulus']['id']
            stimulus_ids.append(stimulus_id)
            if isinstance(self.model, SDCDecoder):
                w, w_pred = self.model(x.to(self.device)[None], y.to(self.device)[None])
                W_pred.append(w_pred.cpu())
                W.append(w.cpu())
            else:
                y_pred = self.model(x.to(self.device)[None])
                y_pred = y_pred.reshape(y.shape)
                Y_pred.append(y_pred.cpu())
            
        if isinstance(self.model, SDCDecoder):
            return torch.stack(Y), torch.stack(W), torch.stack(W_pred), torch.tensor(stimulus_ids), 
        else:
            return torch.stack(Y), torch.stack(Y_pred), torch.tensor(stimulus_ids)
    
    def get_evaluation_subset(
            self,
            dataset: Dataset,
            subset_size: int,
            subset_seed: int
    ):
        N = len(dataset)
        np.random.seed(subset_seed)
        ids = np.random.choice(N, size=subset_size, replace=False)
        ids.sort()
        dataset = Subset(dataset, ids)
        return dataset

    def evaluate(
            self,
            evaluation_metrics: Optional[Sequence[Callable]] = (r2_score, pearsonr),
            distance_metrics: Optional[Sequence[Callable]] = (cosine_distance, mean_squared_distance),
            distance_classification_measures: Optional[Sequence[Callable]] = None,
            subset_size: Optional[int] = None,
            subset_seed: int = 0,
            top_knn_values: Optional[Sequence[int]] = (1, 5, 10, 50, 100, 500)
    ):
        if evaluation_metrics is None:
            evaluation_metrics = []
        if distance_classification_measures is None:
            distance_classification_measures = []
        if distance_metrics is None:
            distance_metrics = []
        evaluation_dict = {}
        folds = [
            ('train', self.train_dataset),
            ('val', self.val_dataset),
        ]
        if subset_size:
            folds = [
                (fold_name, self.get_evaluation_subset(dataset, subset_size, subset_seed))
                for fold_name, dataset in folds
            ]

        self.model.eval()
        for fold_name, dataset in folds:
            with torch.no_grad():
                if isinstance(self.model, SDCDecoder):
                    Y, W, W_pred, stimulus_ids = self.run_all(dataset)
                else:
                    Y, Y_pred, stimulus_ids = self.run_all(dataset)

            #if self.mode == 'decode':
            evaluation_measures = (evaluation_metrics, distance_metrics, distance_classification_measures,
                                    top_knn_values)
            #else:
            #    evaluation_measures = (evaluation_metrics, [], [])
            if isinstance(self.model, SDCDecoder):
                evaluation = evaluate_decoding(W, W_pred, stimulus_ids, fold_name, *evaluation_measures)
            else:
                evaluation = evaluate_decoding(Y, Y_pred, stimulus_ids, fold_name, *evaluation_measures)

            merge_dicts(
                source=evaluation,
                dest=evaluation_dict
            )
        return evaluation_dict

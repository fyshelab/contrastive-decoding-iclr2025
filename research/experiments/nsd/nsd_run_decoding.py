import gc
from typing import Optional, Dict

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, Subset
import numpy as np
import h5py
from pathlib import Path
import wandb
import fire

from research.data.natural_scenes import (
    NaturalScenesDataset,
    KeyDataset
)
from research.models.fmri_decoders import Decoder
from research.metrics.loss_functions import (
    CosineSimilarityLoss,
    ContrastiveDistanceLoss,
    BalancedContrastiveLoss,
)
from research.experiments.nsd.nsd_experiment import NSDExperiment
from research.metrics.metrics import (
    cosine_distance,
    squared_euclidean_distance,
)
from pipeline.utils import product


def run_experiment(
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        channels_last: bool,
        distance_metric: str = 'cosine',
        loss: str = 'distance',
        group: str = None,
        max_iterations: int = 10001,
        evaluation_interval: int = 250,
        notes: str = None,
        config: Dict = None,
        wandb_logging: bool = False,
        sdc_decoder: bool = False,
        sdc_size: int = None,
        sdc_distance_metric: str = 'cosine',
        temperature: float = 1.,
        seed: int = 0,
        mode: str = 'decode',
):
    if config is None:
        config = {}
    device = torch.device('cuda')

    sample = train_dataset[0]
    betas_shape = sample['betas'][0].shape
    stimulus_shape = sample['stimulus']['data'].shape

    torch.manual_seed(seed)

    if sdc_decoder:
        model_params = dict(
            in_size=betas_shape[0],
            sdc_size=sdc_size,
            embedding_size=product(stimulus_shape),
            normalize=(distance_metric == 'cosine'),
        )
        model = SDCDecoder(**model_params)
    else:
        in_size = betas_shape[0]
        out_size = product(stimulus_shape)
        if mode == 'encode':
            in_size, out_size = out_size, in_size
        model_params = dict(
            layer_sizes=[
                in_size,
                #5000,
                out_size,
            ],
            dropout_p=0.0,
            normalize=(distance_metric == 'cosine'),
        )
        model = Decoder(**model_params)

    model.to(device)
    criterion_params = dict(temperature=temperature)
    if loss == 'balanced':
        if distance_metric != 'cosine':
            raise ValueError()
        criterion_params = dict(l2_weight=0, cosine_weight=1, contrastive_weight=1)
        criterion = BalancedContrastiveLoss(**criterion_params)
    elif distance_metric == 'euclidean':
        if loss == 'distance':
            criterion = nn.MSELoss(**criterion_params)
        elif loss == 'contrastive':
            criterion = ContrastiveDistanceLoss(distance_metric=squared_euclidean_distance, temperature=temperature)
    elif distance_metric == 'cosine':
        if loss == 'distance':
            criterion = CosineSimilarityLoss(**criterion_params)
        elif loss == 'contrastive':
            criterion = ContrastiveDistanceLoss(distance_metric=cosine_distance, temperature=temperature)
    else:
        raise ValueError()
    if sdc_decoder:
        if sdc_distance_metric == 'cosine':
            sdc_criterion = ContrastiveDistanceLoss(distance_metric=cosine_distance, temperature=temperature)
        elif sdc_distance_metric == 'euclidean':
            sdc_criterion = ContrastiveDistanceLoss(distance_metric=squared_euclidean_distance, temperature=temperature)
        else:
            raise ValueError()
    else:
        sdc_criterion = None


    optimizer_params = dict(lr=1e-4)
    optimizer = Adam(
        params=model.parameters(),
        **optimizer_params,
    )
    sdc_optimizer = None

    scheduler_params = dict(
        lr_schedule_dict={}#{3: 0.1, 6: 0.01, 9: 0.001}
    )

    def lr_schedule(epoch):
        lr_scale = 1
        for k, v in scheduler_params['lr_schedule_dict'].items():
            if epoch >= k:
                lr_scale = v
        return lr_scale

    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

    evaluation_objective = ('top_knn_accuracy', 'cosine', str(5), 'val')

    training_params = dict(
        batch_size=batch_size,
        evaluation_interval=evaluation_interval,
        evaluation_subset_size=3000,
        evaluation_objective=evaluation_objective,
        distance_metric=distance_metric,
        drop_last=True,
        augmentation=True,
        augmentation_noise_scale=1.0,
    )
    experiment = NSDExperiment(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        sdc_criterion=sdc_criterion,
        mode=mode,
        sdc_optimizer=sdc_optimizer,
        **training_params
    )

    config = {
        **config,
        'model': str(model),
        **model_params,
        'criterion': str(criterion),
        **criterion_params,
        'optimizer': str(optimizer),
        **optimizer_params,
        **scheduler_params,
        **training_params,
    }
    if wandb_logging:
        wandb.init(project='natural-scenes', config=config, group=group, notes=notes)
        wandb.define_metric("*", summary="max")
        wandb.define_metric("*", summary="min")
        experiment.wandb_run = wandb.run

    #try:
    experiment.train_model(max_iterations=max_iterations, logger=wandb.log if wandb_logging else None)
    #except KeyboardInterrupt():
    #    print("Handled keyboard interrupt during training. Returning model.")
    return experiment


def main(
        nsd_path: str,
        subject_name: str,
        model_name: str,
        stimulus_key: str,
        voxel_selection_path='derivatives/voxel-selection.hdf5',
        voxel_selection_key='nc/sorted_indices_flat',
        permutation_test: bool = False,
        permutation_fraction: float = 1.0,
        group: Optional[str] = None,
        num_voxels: Optional[int] = 2500,
        threshold: Optional[float] = None,
        result_key: Optional[str] = None,
        max_iterations: int = 10001,
        loss: str = 'distance',
        batch_size: int = 128,
        save_best_model: bool = False,
        temperature: float = 1.,
        seed: int = 0,
        evaluation_interval: int = 100,
):
    nsd = NaturalScenesDataset(nsd_path)

    experiment_params = dict(
        batch_size=batch_size,
        distance_metric='cosine' if 'embedding' in stimulus_key else 'euclidean',
        group=group,
        max_iterations=max_iterations,
        evaluation_interval=evaluation_interval,
        channels_last=False, # (model_name == 'ViT-B=32' and stimulus_key != 'embedding'),
        wandb_logging=True,
        loss=loss,
        sdc_decoder=False,
        sdc_size=512,
        sdc_distance_metric='cosine',
        temperature=temperature,
        seed=seed,
    )

    betas_params = dict(
        subject_name=subject_name,
        voxel_selection_path=voxel_selection_path,
        voxel_selection_key=voxel_selection_key,
        num_voxels=num_voxels,
        threshold=threshold,
        return_volume_indices=True,
    )
    betas, betas_indices = nsd.load_betas(**betas_params)
    betas_params['num_voxels'] = betas_indices.shape[0]

    if permutation_test:
        indices = np.arange(len(betas))
        shuffle_mask = np.zeros_like(indices, dtype=bool)
        shuffle_mask[:int(shuffle_mask.shape[0] * permutation_fraction)] = True
        np.random.shuffle(shuffle_mask)
        shuffle_ids = indices[shuffle_mask]
        np.random.shuffle(shuffle_ids)
        indices[shuffle_mask] = shuffle_ids
        betas = Subset(betas, list(indices))

    stimulus_params = dict(
        subject_name=subject_name,
        #stimulus_path='nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5',
        #stimulus_key='imgBrick',
        stimulus_path=f'derivatives/stimulus_embeddings/{model_name}.hdf5',
        stimulus_key=stimulus_key,
        delay_loading=True
    )
    stimulus = nsd.load_stimulus(**stimulus_params)

    dataset = KeyDataset({'betas': betas, 'stimulus': stimulus})
    train_dataset, val_dataset, test_dataset = nsd.apply_subject_split(dataset, subject_name, 'split-01')

    config = {'model_name': model_name, **betas_params, **stimulus_params, 'save_best_model': save_best_model, **experiment_params}
    experiment = run_experiment(
        train_dataset,
        val_dataset,
        config=config,
        **experiment_params,
    )
    if save_best_model:
        experiment.model.load_state_dict(experiment.best_state_dict)

    with torch.no_grad():
        if isinstance(experiment.model, SDCDecoder):
            Y_val,  W_val, W_val_pred, Y_val_ids = experiment.run_all(val_dataset)
            Y_test, W_test, W_test_pred, Y_test_ids = experiment.run_all(test_dataset)
        else:
            Y_val, Y_val_pred, Y_val_ids = experiment.run_all(val_dataset)
            Y_test, Y_test_pred, Y_test_ids = experiment.run_all(test_dataset)

    def require_dataset(group, key, tensor):
        if key in group:
            group[key][:] = tensor
        else:
            group[key] = tensor
    results_path = Path(nsd_path) / 'derivatives/decoded_features'

    key_name = wandb.run.group if wandb.run.group else wandb.run.name
    save_file_path = results_path / wandb.config['model_name'] / f'{key_name}.hdf5'
    save_file_path.parent.mkdir(exist_ok=True, parents=True)

    h5_key = [wandb.config['subject_name'], wandb.config['stimulus_key']]
    if result_key is not None:
        if result_key == 'wandb_run_name':
            h5_key += [wandb.run.name]
        else:
            h5_key += [result_key]

    attributes = dict(wandb.config)
    attributes['wandb_run_name'] = wandb.run.name
    attributes['wandb_run_url'] = wandb.run.url
    attributes['wandb_group'] = wandb.run.group
    attributes['wandb_notes'] = wandb.run.notes
    attributes['evaluation_objective'] = experiment.evaluation_objective
    attributes['best_iteration'] = experiment.best_iteration
    attributes['best_score'] = str(experiment.best_score)

    with h5py.File(save_file_path, 'a') as f:
        key = '/'.join(h5_key)
        group = f.require_group(key)
        for k, v in attributes.items():
            if isinstance(v, dict):
                v = str(v)
            group.attrs[k] = v
        group.attrs['iteration'] = experiment.iteration
        require_dataset(group, 'volume_indices', betas_indices)
        require_dataset(group, 'test/stimulus_ids', Y_test_ids)
        require_dataset(group, 'val/stimulus_ids', Y_val_ids)
        if isinstance(experiment.model, SDCDecoder):
            require_dataset(group, 'val/W', W_val)
            require_dataset(group, 'val/W_pred', W_val_pred)
            require_dataset(group, 'test/W', W_test)
            require_dataset(group, 'test/W_pred', W_test_pred)
        else:
            require_dataset(group, 'val/Y_pred', Y_val_pred.detach().cpu())
            require_dataset(group, 'test/Y_pred', Y_test_pred.detach().cpu())
        
        model_group = group.require_group('model')
        for param_name, weights in experiment.model.state_dict().items():
            weights = weights.cpu()
            require_dataset(model_group, param_name, weights)

    wandb.finish()
    torch.cuda.empty_cache()
    gc.collect()
    return experiment

def linear_reruns(
        nsd_path: str,
        subject: str,
        run_id: int,
):
    model_name = 'ViT-B=32'
    stimulus_key = 'embedding'
    voxel_selection = 'nc'
    threshold = 8
    num_voxels = None

    if voxel_selection == 'nc':
        #voxel_selection_path = 'derivatives/noise-ceiling-fsavg-top4.hdf5'
        voxel_selection_path = 'derivatives/noise-ceiling.hdf5'
        if threshold is None:
            voxel_selection_key = f'split-01/sorted_indices'
        else:
            voxel_selection_key = f'split-01/value'
    elif voxel_selection == 'fracridge':
        voxel_selection_path = f'derivatives/encoded_betas/{model_name}/fracridge.hdf5'
        if threshold is None:
            voxel_selection_key = f'{stimulus_key}/volume_indices'
        else:
            voxel_selection_key = f'{stimulus_key}/value'
        
    group_name = 'group-22_reruns'
    seed = run_id + 1

    h5_file_path = Path(nsd_path) / f'derivatives/decoded_features/{model_name}/{group_name}.hdf5'
    if not h5_file_path.exists():
        print(f"{h5_file_path} does not exist")
    else:
        f = h5py.File(h5_file_path, 'r')
        if f'{subject}/{stimulus_key}/run_{run_id}' in f:
            print(f"run_{run_id} {subject} already finished")
            return
        f.close()
        
    print(f"starting run_{run_id} {subject}")

    experiment = main(
        nsd_path,
        subject,
        model_name,
        stimulus_key,
        voxel_selection_path=voxel_selection_path,
        voxel_selection_key=voxel_selection_key,
        group=group_name,
        permutation_test=False,
        #permutation_fraction=permutation_fraction,
        max_iterations=5001,
        #loss='contrastive',
        loss=loss,
        batch_size=128,
        num_voxels=None,
        threshold=8,
        temperature=0.03,
        seed=seed,
        result_key=f'run_{run_id}',
        evaluation_interval=1000,
    )

    print(f"completed run-{run_id} {subject}")
    
def tuning(
        nsd_path: str,
        subject: str,
):
    model_name = 'ViT-B=32'
    stimulus_key = 'embedding'
    voxel_selection = 'nc'
    threshold = 8
    num_voxels = None

    if voxel_selection == 'nc':
        #voxel_selection_path = 'derivatives/noise-ceiling-fsavg-top4.hdf5'
        voxel_selection_path = 'derivatives/noise-ceiling.hdf5'
        if threshold is None:
            voxel_selection_key = f'split-01/sorted_indices'
        else:
            voxel_selection_key = f'split-01/value'
    elif voxel_selection == 'fracridge':
        voxel_selection_path = f'derivatives/encoded_betas/{model_name}/fracridge.hdf5'
        if threshold is None:
            voxel_selection_key = f'{stimulus_key}/volume_indices'
        else:
            voxel_selection_key = f'{stimulus_key}/value'
        
    group_name = 'tuning'

    experiment = main(
        nsd_path,
        subject,
        model_name,
        stimulus_key,
        voxel_selection_path=voxel_selection_path,
        voxel_selection_key=voxel_selection_key,
        group=group_name,
        permutation_test=False,
        #permutation_fraction=permutation_fraction,
        max_iterations=5001,
        #loss='contrastive',
        loss=loss,
        batch_size=128,
        num_voxels=None,
        threshold=8,
        temperature=0.03,
        seed=0,
        result_key=f'wandb_run_name',
        evaluation_interval=1000,
    )

loss = 'contrastive'

def clip(
        nsd_path: str,
        subject: str = 'all',
        model_name: str = 'ViT-B=32',
        decode_resblocks: bool = False,
        permutation_test: bool = False,
        permutation_fraction: float = 1.,
        voxel_selection: str = 'nc',
        num_voxels: Optional[int] = None,
        threshold: Optional[float] = None,
):
    run_models = [
        (model_name, 'embedding'),
    ]
    if decode_resblocks:
        run_models += [(model_name, f'transformer.resblocks.{i}') for i in range(12)]
    if subject == 'all':
        subjects = [f'subj0{i}' for i in range(1, 9)]
    else:
        subjects = [subject]

    for subject_name in subjects:
        for model_name, stimulus_key in run_models:
            if voxel_selection == 'nc':
                voxel_selection_path = 'derivatives/noise-ceiling.hdf5'
                if threshold is None:
                    voxel_selection_key = f'split-01/sorted_indices'
                else:
                    voxel_selection_key = f'split-01/value'
            elif voxel_selection == 'fracridge':
                voxel_selection_path = f'derivatives/encoded_betas/{model_name}/fracridge.hdf5'
                if threshold is None:
                    voxel_selection_key = f'{stimulus_key}/volume_indices'
                else:
                    voxel_selection_key = f'{stimulus_key}/value'
            else:
                raise ValueError()

            main(
                nsd_path,
                subject_name,
                model_name,
                stimulus_key,
                voxel_selection_path=voxel_selection_path,
                voxel_selection_key=voxel_selection_key,
                permutation_test=permutation_test,
                permutation_fraction=permutation_fraction,
                threshold=threshold,
                num_voxels=num_voxels,
            )


if __name__ == '__main__':
    fire.Fire({f.__name__: f for f in [main, clip, linear_reruns, tuning]})

#from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from random import Random
from typing import Callable, Optional, Sequence, Tuple, Dict, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, Subset
import pandas as pd
import h5py
import nibabel as nib
from einops import rearrange
from pycocotools.coco import COCO
from nsdcode.nsd_mapdata import NSDmapdata
import matplotlib.pyplot as plt

from pipeline.utils import index_unsorted, DisablePrints, read_patch

BETAS_SCALE = 300
BETAS_PER_SESSION = 750


class NaturalScenesDataset:
    def __init__(
            self,
            dataset_path: str,
            resolution_name: str = 'func1pt8mm',
            preprocess_name: str = 'betas_fithrf_GLMdenoise_RR',
            coco_path: Optional[str] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.derivatives_path = self.dataset_path / 'derivatives'
        self.ppdata_path = self.dataset_path / 'nsddata' / 'ppdata'

        self.subjects = {f'subj0{i}': {} for i in range(1, 9)}
        for subject_name, subject_data in self.subjects.items():
            responses_file_path = self.ppdata_path / subject_name / 'behav' / 'responses.tsv'
            subject_data['responses'] = pd.read_csv(responses_file_path, sep='\t',)

            # The last 3 sessions are currently held-out for the algonauts challenge
            # remove them for now.
            session_ids = subject_data['responses']['SESSION']
            held_out_mask = session_ids > (np.max(session_ids) - 3)
            subject_data['responses'] = subject_data['responses'][~held_out_mask]

            subject_betas_path = self.derivatives_path / 'betas' / subject_name / resolution_name / preprocess_name
            num_sessions = np.max(subject_data['responses']['SESSION'])

            subject_data['betas'] = h5py.File(subject_betas_path / f'betas_sessions.hdf5', 'r')

            func_path = self.ppdata_path / subject_name / resolution_name
            roi_path = func_path / 'roi'
            derivative_image_path = self.derivatives_path / 'images' / subject_name / resolution_name
            roi_paths = {
                **{name: func_path / f'{name}.nii.gz'
                   for name in ('brainmask', 'aseg', 'hippoSfLabels')},
                **{name: roi_path / f'{name}.nii.gz'
                   for name in ('corticalsulc', 'floc-bodies', 'floc-faces', 'floc-places', 'floc-words',
                                'HCP_MMP1', 'Kastner2015', 'MTL', 'nsdgeneral', 'prf-eccrois', 'prf-visualrois', 'streams', 'thalamus')},
                **{name: derivative_image_path / f'{name}.nii.gz'
                   for name in ('wm', 'aparc', 'aparc.DKTatlas', 'aparc.a2009s', 'HCP_MMP1_cortices')},
            }
            for hemi in ('lh', 'rh'):
                roi_paths.update({
                    f'{hemi}.{name}': path.parent / f'{hemi}.{path.name}'
                    for name, path in roi_paths.items()
                })

            subject_data['roi_paths'] = roi_paths
            label_name_path = self.dataset_path / 'nsddata' / 'freesurfer' / subject_name / 'label'
            ctab_files = [p for p in label_name_path.iterdir() if p.suffix == '.ctab']
            ctab_files += list((self.derivatives_path / 'labels' / subject_name).iterdir())
            subject_data['roi_label_names'] = label_names = {}
            for roi_name in roi_paths.keys():
                for ctab_file in ctab_files:
                    if ctab_file.name.startswith(f'{roi_name}.'):
                        with open(ctab_file) as f:
                            lines = [line.strip().split(' ') for line in f.readlines()]
                            label_names[roi_name] = {int(line[0]): line[1] for line in lines}
                        label_names[roi_name][-1] = 'Unlabeled'

        simulus_information_path = self.dataset_path / 'nsddata' / 'experiments' / 'nsd' / 'nsd_stim_info_merged.csv'
        self.stimulus_info = pd.read_csv(simulus_information_path, index_col=0)

        self.coco_path = None
        if coco_path:
            self.coco_path = Path(coco_path)
            with DisablePrints():
                annotation_path = self.coco_path / 'annotations'
                fold_names = ('train2017', 'val2017')
                annotation_types = ('captions',)
                self.coco_folds = {
                    fold_name: {
                        annotation_type: COCO(annotation_path / f'{annotation_type}_{fold_name}.json')
                        for annotation_type in annotation_types
                    }
                    for fold_name in fold_names
                }

        np.int = int
        self.map_data = NSDmapdata(dataset_path)

        self.lh_flat = read_patch(self.dataset_path / 'nsddata/freesurfer/fsaverage/surf/lh.full.flat.patch.3d')
        self.rh_flat = read_patch(self.dataset_path / 'nsddata/freesurfer/fsaverage/surf/rh.full.flat.patch.3d')

    def get_affine(self, subject_name):
        return nib.load(self.subjects[subject_name]['roi_paths']['brainmask']).affine

    def load_roi(
            self,
            subject_name,
            roi_name,
    ):
        subject = self.subjects[subject_name]
        image = nib.load(subject['roi_paths'][roi_name]).get_fdata()
        image = image.astype(int)
        label_names = {}
        if roi_name in subject['roi_label_names']:
            label_names = subject['roi_label_names'][roi_name]
        return image, label_names

    def load_betas(
            self,
            subject_name: str,
            betas_indices: Optional[np.ndarray] = None,
            voxel_selection_path: Optional[str] = None,
            voxel_selection_key: Optional[str] = None,
            num_voxels: Optional[int] = None,
            threshold: Optional[float] = None,
            return_volume_indices: bool = False,
            return_tensor_dataset: bool = True,
            session_normalize: bool = True,
            scale_betas: bool = True,
    ):
        subject_betas = self.subjects[subject_name]['betas']
        
        if betas_indices is None:
            voxel_selection_file = h5py.File(self.dataset_path / voxel_selection_path, 'r')
            key = f'{subject_name}/{voxel_selection_key}'
            voxel_selection_map = voxel_selection_file[key][:]
            if num_voxels is not None:
                betas_indices = voxel_selection_map[:num_voxels]

            elif threshold is not None:
                betas_indices = np.where(voxel_selection_map.flatten() > threshold)[0]
            else:
                raise ValueError()

        if len(betas_indices.shape) > 1:
            betas_indices = self.flatten_indices(subject_name, betas_indices)

        betas = np.stack([
            subject_betas['betas'][:, i]
            for i in betas_indices
        ], axis=1)
        betas = betas.astype(np.float32)
        if scale_betas:
            betas = betas / BETAS_SCALE

        if session_normalize:
            betas = rearrange(betas, '(s b) v -> s b v', b=BETAS_PER_SESSION)
            betas = (betas - betas.mean(axis=1, keepdims=True)) / (betas.std(axis=1, keepdims=True) + 1e-7)
            betas = rearrange(betas, 's b v -> (s b) v')

        if return_tensor_dataset:
            betas = TensorDataset(torch.from_numpy(betas))

        out = [betas]
        if return_volume_indices:
            volume_indices = subject_betas['indices'][:][betas_indices]
            out.append(volume_indices)
        return tuple(out)

    def load_stimulus(
            self,
            subject_name: Optional[str] = None,
            stimulus_path: str = 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5',
            stimulus_key: str = 'imgBrick',
            delay_loading: bool = False,
            return_tensor_dataset: bool = True,
            return_stimulus_ids: bool = False,
    ):
        stimulus_file = h5py.File(self.dataset_path / stimulus_path, 'r')
        stimulus = stimulus_file[stimulus_key]

        responses = self.subjects[subject_name]['responses']
        response_stimulus_ids = responses['73KID'].to_numpy() - 1

        if delay_loading:
            if not return_tensor_dataset:
                raise RuntimeError()
            return StimulusDataset(stimulus, response_stimulus_ids)
        else:
            stimulus_data = index_unsorted(stimulus, response_stimulus_ids)
            if return_tensor_dataset:
                return TensorDataset(torch.from_numpy(stimulus_data))
            else:
                if return_stimulus_ids:
                    return stimulus_data, response_stimulus_ids
                else:
                    return stimulus_data

    def get_split(
            self,
            subject_name: str,
            split_name: str,
    ):
        split = h5py.File(self.derivatives_path / 'data_splits' / f'{split_name}.hdf5')
        subject_split = split[subject_name]

        test_mask = subject_split['test_response_mask'][:].astype(bool)
        val_mask = subject_split['validation_response_mask'][:].astype(bool)
        train_mask = ~(test_mask | val_mask)
        return train_mask, val_mask, test_mask

    def apply_subject_split(
            self,
            dataset: Dataset,
            subject_name: str,
            split_name: str,
    ):
        train_mask, val_mask, test_mask = self.get_split(subject_name, split_name)
        train_dataset = Subset(dataset, np.where(train_mask)[0])
        val_dataset = Subset(dataset, np.where(val_mask)[0])
        test_dataset = Subset(dataset, np.where(test_mask)[0])
        return train_dataset, val_dataset, test_dataset

    def apply_nfold_split(
            self,
            dataset: Dataset,
            num_folds: int,
            select_fold: int,
            seed: int = 0,
    ):
        assert select_fold < num_folds
        fold_ids = [i % num_folds for i in range(len(dataset))]
        Random(seed).shuffle(fold_ids)
        fold_ids = np.array(fold_ids)
        train_dataset = Subset(dataset, np.where(fold_ids != select_fold)[0])
        test_dataset = Subset(dataset, np.where(fold_ids == select_fold)[0])
        return train_dataset, test_dataset

    def combine_nfold_tensors(
            self,
            tensors: Sequence[torch.Tensor],
            num_folds: int,
            seed: int = 0
    ):
        N = sum(tensor.shape[0] for tensor in tensors)
        fold_ids = [i % num_folds for i in range(N)]
        Random(seed).shuffle(fold_ids)
        fold_ids = torch.tensor(fold_ids)

        inverse_subset_ids = torch.argsort(torch.cat([
            torch.where(fold_ids == select_fold)[0]
            for select_fold in range(num_folds)
        ]))

        return torch.cat(tensors)[inverse_subset_ids]

    def volume_shape(self, subject_name: str):
        if isinstance(subject_name, int):
            subject_name = f'subj0{subject_name + 1}'
        subject_betas = self.subjects[subject_name]['betas']
        volume_shape = tuple(subject_betas['betas'].attrs['spatial_shape'])
        return volume_shape

    def reconstruct_volume(
            self,
            subject_name: str,
            values: Union[float, torch.Tensor],
            indices: torch.Tensor,
            fill_value: Any = 0.,
    ):
        volume_shape = self.volume_shape(subject_name)

        volume = torch.full(volume_shape, fill_value, dtype=values.dtype)
        volume[indices[:, 0], indices[:, 1], indices[:, 2]] = values
        return volume

    def flatten_indices(
            self,
            subject_name: str,
            volume_indices: torch.Tensor
    ):
        subject_betas = self.subjects[subject_name]['betas']
        H, W, D = tuple(subject_betas['betas'].attrs['spatial_shape'])
        volume = np.arange(H * W * D).reshape((H, W, D))
        flat_indices = volume[volume_indices[:, 0], volume_indices[:, 1], volume_indices[:, 2]]
        return flat_indices

    def load_decoder(
            self, 
            subject_id: int,
            model_name: str, 
            group_name: str,
            embedding_name: str,
            model_class: torch.nn.Module,
    ):
        subject_name = f'subj0{subject_id + 1}'
        embeddings = h5py.File(self.derivatives_path / f'decoded_features/{model_name}/{group_name}.hdf5', 'r')
        subject_embeddings = embeddings[f'{subject_name}/{embedding_name}']
        config = dict(subject_embeddings.attrs)

        model_params = {k: config[k] for k in ('layer_sizes', 'dropout_p')}
        model = model_class(**model_params)
        model = model.eval()
        state_dict = {k: torch.from_numpy(v[:]) for k, v in subject_embeddings['model'].items()}
        model.load_state_dict({k: v.clone() for k, v in state_dict.items()})

        betas_params = {
            k: config[k] 
            for k in (
                'subject_name', 'voxel_selection_path', 
                'voxel_selection_key', 'num_voxels', 'return_volume_indices', 'threshold'
            )
        }
        if betas_params['threshold'] is not None:
            betas_params['num_voxels'] = None
            betas_params['return_tensor_dataset'] = False
        X, volume_indices = self.load_betas(**betas_params)

        stimulus_params = dict(
            subject_name=subject_name,
            stimulus_path=f'derivatives/stimulus_embeddings/{model_name}.hdf5',
            stimulus_key=embedding_name,
            delay_loading=False,
            return_tensor_dataset=False,
            return_stimulus_ids=True,
        )
        Y, stimulus_ids = self.load_stimulus(**stimulus_params)
        Y = Y.astype(np.float32)

        Y_pred_val = subject_embeddings[f'val/Y_pred'][:]
        Y_pred_val = Y_pred_val / np.linalg.norm(Y_pred_val, axis=1)[:, None]
        Y_pred_test = subject_embeddings[f'test/Y_pred'][:]
        Y_pred_test = Y_pred_test / np.linalg.norm(Y_pred_test, axis=1)[:, None]

        return model, X, volume_indices, Y, stimulus_ids, Y_pred_val, Y_pred_test


    def load_coco(self, i):
        stim_info = dict(self.stimulus_info.loc[i])
        coco_stim_id = stim_info['cocoId']
        fold = self.coco_folds[stim_info['cocoSplit']]
        #coco = fold['instances']
        #image_info = coco.loadImgs([coco_stim_id])[0]

        coco_captions = fold['captions']
        annotation_ids = coco_captions.getAnnIds(imgIds=coco_stim_id)
        captions = [
            annotation['caption']
            for annotation in coco_captions.loadAnns(annotation_ids)
        ]
        return captions

    def to_fs_subject_space(
            self,
            subject_id: int,
            source_data: np.ndarray,
            source_space: str = 'func1pt8',
            target_space: str = 'pial',
            interp_type: str = 'cubic',
    ):
        return {
            hemi: self.map_data.fit(
                subject_id + 1,
                sourcespace=source_space,
                targetspace=f'{hemi}.{target_space}',
                sourcedata=source_data,
                interptype=interp_type,
            )
            for hemi in ['lh', 'rh']
        }

    def to_fs_average_space(
            self,
            subject_id: int,
            source_data: Union[np.ndarray, Dict],
            source_space: str = 'func1pt8',
            subject_space: str = 'pial',
            interp_type: str = 'cubic',
    ):
        volume_spaces = ['anat0pt5', 'anat0pt8', 'anat1pt0', 'func1pt0', 'func1pt8', 'MNI']
        if source_space in volume_spaces:
            subject_data = self.to_fs_subject_space(
                subject_id=subject_id, source_data=source_data, source_space=source_space, target_space=subject_space, interp_type=interp_type
            )
        elif source_space == 'fssubject':
            subject_data = source_data
        else:
            raise ValueError()
        
        return {
            hemi: self.map_data.fit(
                subject_id + 1,
                sourcespace=f'{hemi}.white',
                targetspace='fsaverage',
                sourcedata=data,
                interptype=interp_type
            )
            for hemi, data in subject_data.items()
        }

    def to_func_space(
            self,
            subject_id: int,
            source_data: Dict[str, np.ndarray],
            res: Tuple[int, int, int],
            target_space: str = 'func1pt8',
            interp_type: str = 'linear',
    ):
        source_data = {
            space: self.map_data.fit(
                subject_id + 1,
                targetspace=f'{space[:2]}.white',
                sourcespace='fsaverage',
                sourcedata=data,
                interptype=interp_type
            )
            for space, data in source_data.items()
        }

        lh_source_data = {k: v for k, v in source_data.items() if k.startswith('lh')}
        rh_source_data = {k: v for k, v in source_data.items() if k.startswith('rh')}

        for k, v in rh_source_data.items():
            print(k, v.shape)

        lh_out = self.map_data.fit(
            subject_id + 1,
            sourcespace=list(lh_source_data.keys()),
            targetspace=target_space,
            sourcedata=np.array(list(lh_source_data.values())).T,
            interptype=interp_type,
            res=res,
        )
        rh_out = self.map_data.fit(
            subject_id + 1,
            sourcespace=list(rh_source_data.keys()),
            targetspace=target_space,
            sourcedata=np.array(list(rh_source_data.values())).T,
            interptype=interp_type,
            res=res,
        )

        out = lh_out + rh_out
        overlap = (lh_out != 0) & (rh_out != 0)
        out[overlap] = out[overlap] * 0.5
        return out

    def flat_scatter_plot(
            self,
            lh_data: np.ndarray,
            rh_data: np.ndarray,
            bottomleft_text: str = None,
            bottomright_text: str = None,
            cmap: str = 'jet',
            vmin: float = 0.0,
            vmax: float = 0.5,
            alpha: float = 0.5,
            mask_value: float = None,
            mask_color: str = 'gray'
    ):
        spread = 140
        y_diff = np.max(self.rh_flat['y']) - np.max(self.lh_flat['y'])
        x = np.concatenate([self.lh_flat['x'] - spread, self.rh_flat['x'] + spread])
        y = np.concatenate([self.lh_flat['y'], self.rh_flat['y'] - y_diff])
        c = np.concatenate([lh_data[self.lh_flat['vno']], rh_data[self.rh_flat['vno']]])

        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        xsize = xmax - xmin
        ysize = ymax - ymin

        scale = 0.05
        plt.figure(figsize=(xsize * scale, ysize * scale))
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        padding=5
        fontsize=25
        plt.text(xmin+padding, ymin+padding, bottomleft_text,
                 horizontalalignment='left', verticalalignment='bottom', fontsize=fontsize)
        plt.text(xmax-padding, ymin+padding, bottomright_text,
                 horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize)
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        if mask_value:
            mask = c == mask_value
            plt.scatter(x[mask], y[mask], s=5, c=mask_color, alpha=alpha)
            plt.scatter(x[~mask], y[~mask], s=5, c=c[~mask], cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
        else:
            plt.scatter(x, y, s=5, c=c, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)


class StimulusDataset(Dataset):
    def __init__(self, stimulus: h5py.Dataset, stimulus_ids: torch.Tensor):
        super().__init__()
        self.stimulus = stimulus
        self.stimulus_ids = stimulus_ids

    def __len__(self):
        return self.stimulus_ids.shape[0]

    def __getitem__(self, index):
        stimulus_id = self.stimulus_ids[index]
        stimulus = self.stimulus[stimulus_id]
        return {'data': torch.tensor(stimulus).float(), 'id': stimulus_id}


class KeyDataset(Dataset):
    def __init__(self, datasets: Dict[str, Dataset]):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        keys = list(self.datasets.keys())
        return len(self.datasets[keys[0]])

    def __getitem__(self, index):
        return {
            key: dataset[index]
            for key, dataset in self.datasets.items()
        }


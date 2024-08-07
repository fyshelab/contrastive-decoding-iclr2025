from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
from scipy.cluster import hierarchy
from einops import rearrange
import PIL
from PIL import Image
from sklearn.neighbors import NearestNeighbors


subjects = [f'subj0{i + 1}' for i in range(8)]


def load_decoder_data(
        nsd, 
        subject_id: int,
        model_name: str, 
        group_name: str,
        embedding_name: str,
        model_class: torch.nn.Module,
):
    subject_name = f'subj0{subject_id + 1}'
    embeddings = h5py.File(nsd.derivatives_path / f'decoded_features/{model_name}/{group_name}.hdf5', 'r')
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
    X, volume_indices = nsd.load_betas(**betas_params)

    stimulus_params = dict(
        subject_name=subject_name,
        stimulus_path=f'derivatives/stimulus_embeddings/{model_name}.hdf5',
        stimulus_key=embedding_name,
        delay_loading=False,
        return_tensor_dataset=False,
        return_stimulus_ids=True,
    )
    Y, stimulus_ids = nsd.load_stimulus(**stimulus_params)
    Y = Y.astype(np.float32)

    Y_pred_val = subject_embeddings[f'val/Y_pred'][:]
    Y_pred_val = Y_pred_val / np.linalg.norm(Y_pred_val, axis=1)[:, None]
    Y_pred_test = subject_embeddings[f'test/Y_pred'][:]
    Y_pred_test = Y_pred_test / np.linalg.norm(Y_pred_test, axis=1)[:, None]

    return model, X, volume_indices, Y, stimulus_ids, Y_pred_val, Y_pred_test


def plot_mask_comparisons(run_path, component_ids, reorder_images_path=None):
    num_components = component_ids.shape[0]

    y_all = []
    for subject_id, subject_name in enumerate(subjects):
        M = np.load(run_path / subject_name / 'mask.npy')
        M = M[component_ids]
        M[M != 0] = 1

        M_triu_indices = torch.triu_indices(M.shape[0], M.shape[0], offset=1)
        M_dice = 2 * (M @ M.T) / (M[:, None] + M[None, :]).sum(axis=2)
        M_dice_triu = M_dice[M_triu_indices[0], M_triu_indices[1]]
        y = 1 - M_dice_triu
        y_all.append(y)

    y_all = np.stack(y_all).mean(axis=0)
    Z = hierarchy.linkage(y_all, optimal_ordering=True)
    leaves = hierarchy.leaves_list(Z)

    out_path = run_path / 'mask_comparisons'
    out_path.mkdir(exist_ok=True, parents=True)

    def plot(M_dice, out_file_path):
        plt.figure(figsize=(1.25 * width, width))
        plt.imshow(M_dice, vmax=dice_max)
        plt.yticks(ticks=np.arange(leaves.shape[0]), labels=component_ids[leaves])
        plt.xticks(rotation=60, ha='right', rotation_mode='anchor', ticks=np.arange(num_components), labels=component_ids[leaves])
        plt.colorbar()

        plt.savefig(out_file_path, bbox_inches='tight')
        plt.close()

    M_dice_all = []
    for subject_id, subject_name in enumerate(subjects):
        M = np.load(run_path / subject_name / 'mask.npy')
        M = M[component_ids]
        M[M != 0] = 1

        M_ordered = M[leaves]
        M_dice = 2 * (M_ordered @ M_ordered.T) / (M_ordered[:, None] + M_ordered[None, :]).sum(axis=2)
        M_dice_all.append(M_dice)
        dice_max = np.max(M_dice[M_triu_indices[0], M_triu_indices[1]])

        width = leaves.shape[0] * 0.2
        plot(M_dice, run_path / f'mask_comparisons/{subject_name}.png')
    M_dice_avg = np.stack(M_dice_all).mean(axis=0)

    plot(M_dice_avg, run_path / 'mask_comparisons/averaged.png')

    new_file_path = Path(f'{str(reorder_images_path)}_similarity')
    new_file_path.mkdir(exist_ok=True, parents=True)
    for i, component_id in enumerate(component_ids[leaves]):
        for file_path in reorder_images_path.iterdir():
            if f'component-{component_id}.png' in file_path.name:
                new_file_name = f'component_similarity-{i}_{file_path.name}'
                shutil.copy(file_path, new_file_path / new_file_name)
                continue

    #if reorder_images_path:

def tsne_image_plot(
        y: np.ndarray, 
        stimulus_ids: np.ndarray, 
        stimulus_images: np.ndarray, 
        image_size: int, 
        num_images: int, 
        extent: int
):
    S = image_size * num_images
    full_image = np.zeros(shape=(S, S, 3), dtype=np.ubyte)

    coords = np.linspace(-extent, extent, num_images)
    grid = np.stack(np.meshgrid(coords, coords))
    grid = rearrange(grid, 'd h w -> (h w) d')

    neighbors = NearestNeighbors(metric='chebyshev')
    neighbors.fit(y)

    distances, ids = neighbors.kneighbors(grid, n_neighbors=1,)
    distances = rearrange(distances, '(h w) d -> h w d', h=num_images)
    ids = rearrange(ids, '(h w) d -> h w d', h=num_images)

    distance_threshold = extent / num_images
    for i in range(num_images):
        for j in range(num_images):
            if distances[i, j] > distance_threshold:
                continue
            stimulus_id = stimulus_ids[ids[i, j, 0]]
            stim_image = stimulus_images[stimulus_id]
            stim_image = Image.fromarray(stim_image)
            stim_image = stim_image.resize(size=(image_size, image_size), resample=PIL.Image.LANCZOS)
            stim_image = np.array(stim_image)
            full_image[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size] = stim_image
    return full_image
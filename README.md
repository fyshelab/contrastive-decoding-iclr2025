# Instructions to Reproduce Results
- Download the NSD dataset. You will need the `func1pt8mm/betas_fithrf_GLMdenoise_RR` preparation of the beta weights.
- Clone this repo, create a python environment, and install the project requirements with `pip install -r requirements.txt`
- Copy the `labels` directory in this repo to `nsd/derivatives/labels`
- Run all cells in `research/notebooks/nsd-data_preparation.hdf5`. This will generate the train/val/test split, noise ceiling estimates, and CLIP image embeddings for all stimulus images.
- Train decoder models. Decoders can be trained in series using `research/notebooks/nsd-data_preparation.hdf5` or use gnu parallel to parallelize training on multiple GPUs:

```
parallel --jobs 3 CUDA_VISIBLE_DEVICES='$(({%} + 1))' python -m research.experiments.nsd.nsd_run_decoding linear_reruns \
--nsd_path "~/NSD" \
--subject subj0{} ::: {1..8} \
--run_id {} ::: {0..50}
```
- The ridge regression baseline decoder can also be trained in the last cell of `research/notebooks/nsd-data_preparation.hdf5`
- Run DBSCAN clustering analysis in `research/notebooks/nsd-evaluate_decoding.hdf5`
- Further instructions coming soon
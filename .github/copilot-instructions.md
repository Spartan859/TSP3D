# Copilot instructions for TSP3D

## Build, test, and lint commands

### Environment and build
```bash
conda env create -f environment.yml --name TSP3D
conda activate TSP3D
bash init.sh
```

`init.sh` builds the custom `pointnet2` CUDA extension used by the model code.

If you work on refine/segmentation code paths, build RoI-aware pooling as well:
```bash
cd models/pool3d && python setup.py install --user
cd ../pool3d_4090 && python setup.py install --user
```

### Training
```bash
sh scripts/train_scanrefer_single.sh
sh scripts/train_sr3d.sh
sh scripts/train_nr3d.sh
```

### Evaluation
```bash
sh scripts/test_scanrefer_single.sh
sh scripts/test_sr3d.sh
sh scripts/test_nr3d.sh
```

### Single-test / targeted check
Run the PointNet2 CUDA op gradient test:
```bash
python pointnet2/pointnet2_test.py
# or
pytest pointnet2/pointnet2_test.py::test_interpolation_grad
```

### Lint
There is no repository-standard lint/format command wired in scripts or config files.

## High-level architecture

- **Single entrypoint for train + eval**: `train_dist_mod.py` drives both modes through `TrainTester`, with mode switched by `--eval`.
- **Data pipeline**: `src/joint_det_dataset.py` loads ScanRefer / SR3D / NR3D and can mix in ScanNet detection samples when `--joint_det` is enabled.
- **GT conversion layer**: `get_gt.py` converts dataset tensors to `DepthInstance3DBoxes` structures used by the model heads.
- **Model assembly**: `models/bdetr.py` (`BeaUTyDETR`) combines:
  - Minkowski sparse 3D backbone (`models/mink_resnet.py`, `TSPBackbone`)
  - frozen local RoBERTa encoder
  - multilevel sparse head (`models/multilevel_head_refine.py`, class `TSPHead`)
- **Core head logic**:
  - keep branch: language-guided sparse voxel pruning
  - completion branch: language-guided voxel add-back
  - optional segmentation branch
  - optional refine branch using RoI-aware pooling (`models/roiaware_pool3d_utils.py`)
- **Metrics**: `src/grounding_evaluator.py` reports IoU@0.25/0.5 and subset breakdowns (easy/hard, unique/multi, view-dependent), plus mask metrics when segmentation is enabled.

## Key conventions in this codebase

- `--data_root` is expected to contain dataset assets and `roberta-base/` for offline tokenizer/model loading; many paths are string-concatenated and assume trailing `/`.
- Even single-GPU runs are launched with distributed entrypoints (`torch.distributed.launch` or `torchrun`) in scripts.
- Enabling `--joint_det` implicitly adds ScanNet detection data into training with heavier sampling (`dataset_dict['scannet'] = 10` in `train_dist_mod.py`).
- Scripts commonly switch ScanRefer data scale (`mini`/`large`) by symlinking `ScanRefer_filtered_*` and `*_v3scans.pkl` files before launch.
- RoI-aware pooling backend selection is GPU-aware (`models/roiaware_pool3d_utils.py`) and can be overridden via `POOL3D_VARIANT`.

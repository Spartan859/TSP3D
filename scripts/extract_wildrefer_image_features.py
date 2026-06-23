"""Extract WildRefer image feature maps for projected point fusion."""

import argparse
import json
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/')
    parser.add_argument('--dataset', choices=['strefer', 'liferefer'], default='strefer')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--arch', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def get_wildrefer_root(data_root, dataset):
    source_dir = {'strefer': 'STRefer', 'liferefer': 'LifeRefer'}[dataset]
    candidates = [
        os.path.join(data_root, 'WildRefer', 'src', source_dir),
        os.path.join(data_root, 'WildRefer', source_dir),
        os.path.join(data_root, source_dir),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f'WildRefer source root not found. Checked: {candidates}')


def find_annotation_file(data_root, dataset, split):
    file_name = f'{dataset}_{split}.json'
    candidates = [
        os.path.join(data_root, 'WildRefer', file_name),
        os.path.join(data_root, 'WildRefer', 'downloads', 'json', file_name),
        os.path.join(data_root, file_name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f'WildRefer annotation not found. Checked: {candidates}')


def collect_images(data_root, dataset, splits):
    root = get_wildrefer_root(data_root, dataset)
    items = {}
    for split in splits:
        with open(find_annotation_file(data_root, dataset, split)) as f:
            annos = json.load(f)
        for anno in annos:
            scene_id = str(anno['scene_id'])
            image_name = str(anno['image']['image_name'])
            key = f'{dataset}/{scene_id}/{image_name}'
            for suffix in ('.jpg', '.png'):
                image_path = os.path.join(root, 'image', scene_id, f'{image_name}{suffix}')
                if os.path.exists(image_path):
                    items[key] = image_path
                    break
            if key not in items:
                raise FileNotFoundError(f'Image not found for key={key}')
    return sorted(items.items())


def build_model(arch, pretrained, checkpoint):
    backbone = getattr(models, arch)(pretrained=pretrained)
    layers = [
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
    ]
    model = nn.Sequential(*layers)
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        clean_state = {}
        for key, value in state.items():
            key = key.replace('module.', '')
            if key.startswith('backbone.'):
                key = key[len('backbone.'):]
            clean_state[key] = value
        missing, unexpected = backbone.load_state_dict(clean_state, strict=False)
        print(f'Loaded checkpoint: {checkpoint}')
        print(f'Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')
    elif pretrained:
        print(f'Using torchvision ImageNet pretrained weights for {arch}.')
    else:
        print('WARNING: no --checkpoint provided; image features use random torchvision initialization.')
    model.eval()
    return model


def collate_images(batch, transform):
    keys, paths = zip(*batch)
    images = []
    sizes = []
    for path in paths:
        image = Image.open(path).convert('RGB')
        sizes.append(image.size)
        images.append(transform(image))
    return keys, torch.stack(images, dim=0), sizes


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        args.device = 'cpu'

    items = collect_images(args.data_root, args.dataset, args.splits)
    if args.max_images > 0:
        items = items[:args.max_images]
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = build_model(args.arch, args.pretrained, args.checkpoint).to(args.device)

    mode = 'w' if args.overwrite else 'a'
    with h5py.File(args.output, mode) as h5:
        h5.attrs['arch'] = args.arch
        h5.attrs['image_size'] = args.image_size
        h5.attrs['feature_dim'] = 256 if args.arch in ('resnet18', 'resnet34') else 1024
        pending = [(key, path) for key, path in items if args.overwrite or key not in h5]
        for start in tqdm(range(0, len(pending), args.batch_size), desc='extract'):
            batch = pending[start:start + args.batch_size]
            keys, images, sizes = collate_images(batch, transform)
            images = images.to(args.device, non_blocking=True)
            with torch.no_grad():
                feats = model(images).cpu().numpy().astype(np.float16)
            for key, feat, size in zip(keys, feats, sizes):
                if key in h5:
                    del h5[key]
                dset = h5.create_dataset(key, data=feat, compression='gzip')
                dset.attrs['image_width'] = size[0]
                dset.attrs['image_height'] = size[1]
        h5.attrs['num_images'] = len(items)
    print(f'Saved {len(items)} image feature maps to {args.output}')


if __name__ == '__main__':
    main()

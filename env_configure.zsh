#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")"

export MAMBA_ROOT_PREFIX=/mnt/share/micromamba/root
export CONDA_PKGS_DIRS=/mnt/share/micromamba/pkgs
MICRO=/mnt/share/micromamba/bin/micromamba

mkdir -p /mnt/share/micromamba/bin /mnt/share/micromamba/root /mnt/share/micromamba/pkgs /mnt/share/micromamba/tmp

if [[ ! -x "$MICRO" ]]; then
  curl -LfsS https://micro.mamba.pm/api/micromamba/linux-64/latest -o /mnt/share/micromamba/tmp/micromamba.tar.bz2
  tar -xjf /mnt/share/micromamba/tmp/micromamba.tar.bz2 -C /mnt/share/micromamba/tmp
  install -m 755 /mnt/share/micromamba/tmp/bin/micromamba "$MICRO"
fi

# 启用 clash 代理（用于访问 GitHub / PyPI）
set +u
if command -v clashctl >/dev/null 2>&1; then
  clashctl on || true
else
  source /root/clashctl/scripts/cmd/clashctl.sh
  clashon || true
  clashproxy on || true
fi
set -u

# 配置 micromamba 镜像源（优先 TUNA）
"$MICRO" config remove-key channels || true
"$MICRO" config append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
"$MICRO" config append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
"$MICRO" config append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
"$MICRO" config append channels nvidia
"$MICRO" config set channel_priority flexible
"$MICRO" config set show_channel_urls true
"$MICRO" config remove-key pkgs_dirs || true
"$MICRO" config append pkgs_dirs /mnt/share/micromamba/pkgs

if ! "$MICRO" env list | awk '{print $1}' | grep -qx TSP3D; then
  "$MICRO" create -y -n TSP3D \
    python=3.9 pip cudatoolkit=11.1
fi

"$MICRO" install -y -n TSP3D openblas

# Use Aliyun PyPI mirror for faster pip installs
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export PIP_TRUSTED_HOST=mirrors.aliyun.com
"$MICRO" run -n TSP3D python -m pip install -U pip setuptools wheel -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"
# Install PyTorch (CUDA 11.1) wheels from official PyTorch index (keep official wheel finder)
"$MICRO" run -n TSP3D python -m pip install -f https://download.pytorch.org/whl/torch_stable.html \
  torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"

# Install minimal requirements via Aliyun mirror (reduced to avoid dependency conflicts)
"$MICRO" run -n TSP3D python -m pip install -r requirements-minimal.txt -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"
# Force-install full requirements from requirements.txt ignoring dependency resolution conflicts
# This installs package wheels without resolving/installing their dependencies (--no-deps).
# After this, reinstall typing-extensions to the exact version requested by requirements.txt.
"$MICRO" run -n TSP3D python -m pip install --no-deps -r requirements.txt -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" || true
"$MICRO" run -n TSP3D python -m pip install --force-reinstall typing-extensions==4.5.0 -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" || true
# Note: This will ignore dependency conflicts; some packages may fail at import if required deps are missing or incompatible. Please run pip check inside the env to inspect issues.


"$MICRO" run -n TSP3D python -m pip install openmim==0.3.9
"$MICRO" run -n TSP3D mim install -y mmengine==0.10.4
"$MICRO" run -n TSP3D mim install -y mmcv==2.1.0
"$MICRO" run -n TSP3D mim install -y mmdet==3.2.0
"$MICRO" run -n TSP3D mim install -y mmdet3d==1.4.0

# Ensure numpy <2 for binary compatibility before building native extensions
"$MICRO" run -n TSP3D python -m pip install -U "numpy<2" -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"

ME_INCLUDE="$MAMBA_ROOT_PREFIX/envs/TSP3D/include"

# Rebuild pointnet2 to ensure it is compiled against the pinned numpy/torch
"$MICRO" run -n TSP3D python -m pip uninstall -y pointnet2 || true
"$MICRO" run -n TSP3D python -m pip install -e ./pointnet2 -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" || \
  (cd pointnet2 && "$MICRO" run -n TSP3D python setup.py install)


# Install pointnet2 from local source (editable) so native CUDA ops are built into the env
"$MICRO" run -n TSP3D python -m pip install -e ./pointnet2 || \
  (cd pointnet2 && "$MICRO" run -n TSP3D python setup.py install)

"$MICRO" run -n TSP3D python -m pip install -U \
  git+https://github.com/NVIDIA/MinkowskiEngine \
  --no-deps --no-build-isolation \
  --config-settings=--build-option=--blas_include_dirs="${ME_INCLUDE}" \
  --config-settings=--build-option=--blas=openblas

# Build/install models/pool3d C extension (in-place or editable)
if [ -d "./models/pool3d" ]; then
  echo "Building models/pool3d extension..."
  # Prefer editable install (no-deps) to trigger build_ext; fallback to setup.py build_ext --inplace
  "$MICRO" run -n TSP3D python -m pip install --no-deps -e ./models/pool3d -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" || \
    (cd models/pool3d && "$MICRO" run -n TSP3D python setup.py build_ext --inplace)
fi

"$MICRO" run -n TSP3D python - <<'PY'
import os
# Import checks are best-effort; print exceptions rather than failing the script
def safe_import(name):
    try:
        m=__import__(name)
        v=getattr(m,'__version__',None)
        print(name, 'OK', v)
    except Exception as e:
        print(name, 'FAIL', type(e).__name__, str(e)[:200])

safe_import('torch')
try:
    import torchvision
    print('torchvision', torchvision.__version__)
except Exception as e:
    print('torchvision FAIL', type(e).__name__, str(e)[:200])

safe_import('spacy')
safe_import('mmcv')
safe_import('mmengine')
safe_import('mmdet')
safe_import('mmdet3d')
safe_import('MinkowskiEngine')
# Attempt to import pool3d module if present
try:
    import models.pool3d as _p
    print('models.pool3d OK')
except Exception:
    try:
        import pool3d as _p
        print('pool3d OK')
    except Exception as e:
        print('pool3d FAIL', type(e).__name__, str(e)[:200])

print('python', os.sys.version.split()[0])
print('env_prefix', os.environ.get('CONDA_PREFIX'))
PY

echo "Environment setup finished."

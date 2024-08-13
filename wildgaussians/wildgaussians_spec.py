from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nerfbaselines.registry import MethodSpec
else:
    MethodSpec = dict


WildGaussiansMethodSpec: MethodSpec = {
    "method": "wildgaussians.method:WildGaussians",
    "conda": {
        "environment_name": "wild-gaussians",
        "python_version": "3.11",
        "install_script": r"""
git clone https://github.com/jkulhanek/wild-gaussians.git
cd wild-gaussians
conda install -y --override-channels -c nvidia/label/cuda-11.8.0 cuda-toolkit
if [ "$NB_DOCKER_BUILD" != "1" ]; then
conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
fi
pip install --upgrade pip
pip install -r requirements.txt
LIBRARY_PATH="$CONDA_PREFIX/lib/stubs" pip install -e ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
pip install -e .
"""
    },
    "dataset_overrides": {
        "phototourism": { "config": "phototourism.yml" },
        "nerfonthego": { "config": "nerfonthego.yml" },
        "nerfonthego-undistorted": { "config": "nerfonthego.yml" },
    },
}

try:
    from nerfbaselines.registry import register
    register(WildGaussiansMethodSpec, name="wild-gaussians")
except ImportError:
    pass

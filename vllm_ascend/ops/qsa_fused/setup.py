import os

import torch_npu  # noqa: F401  (registers the PrivateUse1 backend)
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension

ASCEND_TOOLKIT_HOME = os.getenv("ASCEND_TOOLKIT_HOME", os.getenv("ASCEND_HOME_PATH", "null"))
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

ext = NpuExtension(
    name="qsa_fused_ops_lib",
    sources=[os.path.join(BASE_DIR, "qsa_fused_bind.cpp")],
    include_dirs=[
        BASE_DIR,
        os.path.join(ASCEND_TOOLKIT_HOME, "include"),
        os.path.join(ASCEND_TOOLKIT_HOME, "runtime/include"),
    ],
)

setup(
    name="qsa_fused_ops",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)

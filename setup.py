from pathlib import Path
import os
import shutil

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "src" / "csrc"
EXTENSIONS_DIR = ROOT / "build" / "extensions"

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")


class BuildExtensions(BuildExtension):
    def run(self):
        super().run()
        EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
        for ext in self.extensions:
            built_path = Path(self.get_ext_fullpath(ext.name)).resolve()
            if built_path.exists():
                shutil.copy2(built_path, EXTENSIONS_DIR / built_path.name)


setup(
    ext_modules=[
        CUDAExtension(
            name="qwen36_fp8",
            sources=[
                str(CSRC / "fp8_linear.cpp"),
                str(CSRC / "fp8_linear_cuda.cu"),
            ],
            libraries=["cublas"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtensions},
)

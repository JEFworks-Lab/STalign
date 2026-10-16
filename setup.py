"""Build script for STalign.

STalign is pure Python plus one OPTIONAL C++ extension, ``STalign._stalign_cpp``
(sources in STalign/_cpp/), a fast CPU engine for LDDMM and LDDMM_3D_to_slice.
Compiling it needs a C++17 compiler and pybind11 (a build-time dependency only,
see pyproject.toml).  If either is missing, or compilation fails, the build only
warns and STalign is installed without it; STalign then uses its pure-torch
backend, which computes the same thing more slowly.

Development build of the extension inside the source tree:

    python setup.py build_ext --inplace
"""
import re
import sys

from setuptools import find_packages, setup

# the version is defined once, as STalign.__version__
with open('STalign/__init__.py') as f:
    version = re.search(r"^__version__ = '([^']+)'$", f.read(), re.M).group(1)

with open('requirements.txt') as f:
    required = [line.strip() for line in f if line.strip() and not line.startswith('#')]

ext_modules = []
cmdclass = {}
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print('WARNING: pybind11 is not installed; building STalign without its C++ engine '
          '(the pure-torch backend will be used).', file=sys.stderr)
else:
    class BuildExt(build_ext):
        """Optimisation flags per compiler.

        IEEE floating point semantics are required (no -ffast-math), and the
        default target architecture is kept (no -march=native) so the binary is
        portable.  Threads are std::thread (no OpenMP).
        """

        def build_extensions(self):
            if self.compiler.compiler_type == 'msvc':
                flags, link = ['/O2'], []
            else:
                flags, link = ['-O3', '-fno-fast-math'], []
                # Python's own CFLAGS add -fno-strict-overflow (-fwrapv).  The engine does not
                # rely on wrapping signed overflow, and it is ~8% faster without it.
                for attr in ('compiler_so', 'compiler_so_cxx'):
                    cmd = getattr(self.compiler, attr, None)
                    if isinstance(cmd, list):
                        setattr(self.compiler, attr,
                                [f for f in cmd if f not in ('-fwrapv', '-fno-strict-overflow')])
                if sys.platform.startswith('linux'):
                    flags.append('-pthread')
                    link.append('-pthread')
            for ext in self.extensions:
                ext.extra_compile_args += flags
                ext.extra_link_args += link
            super().build_extensions()

    ext_modules.append(Pybind11Extension(
        'STalign._stalign_cpp',
        ['STalign/_cpp/module.cpp'],
        include_dirs=['STalign/_cpp'],
        depends=['STalign/_cpp/engine.hpp', 'STalign/_cpp/fft.hpp', 'STalign/_cpp/pool.hpp',
                 'STalign/_cpp/third_party/pocketfft/pocketfft_hdronly.h'],
        cxx_std=17,
        # a failed compilation is a warning, not an error: STalign falls back to torch
        optional=True,
    ))
    cmdclass['build_ext'] = BuildExt

setup(
    name='STalign',
    author='JEFworks',
    version=version,
    description='aligning spatial transcriptomics data',
    # the C++ engine compiles in pocketfft, whose BSD license must accompany binary distributions
    license_files=['LICENSE', 'STalign/_cpp/third_party/pocketfft/LICENSE.md'],
    packages=find_packages(include=['STalign', 'STalign.*']),
    python_requires='>=3.10',
    install_requires=required,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)

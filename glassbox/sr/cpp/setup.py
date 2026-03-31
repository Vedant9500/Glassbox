from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import sys
from textwrap import dedent

HERE = os.path.dirname(os.path.abspath(__file__))

# Determine OpenMP arguments
openmp_compile_args = ['/openmp'] if os.name == 'nt' else ['-fopenmp', '-O3', '-march=native']
openmp_link_args = [] if os.name == 'nt' else ['-fopenmp']

ext_modules = [
    Pybind11Extension(
        "_core",  # Module name (build in current directory)
        [os.path.join(HERE, "core.cpp")],  # Absolute path so it resolves regardless of CWD
        include_dirs=[os.path.join(HERE, "eigen")],
        extra_compile_args=(["/O2"] if os.name == 'nt' else []) + openmp_compile_args,
        extra_link_args=openmp_link_args,
    ),
]


def _print_project_help() -> None:
    script = os.path.basename(__file__)
    print(
        dedent(
            f"""
            Glassbox SR C++ extension build helper

            Usage:
              python {script} [command] [options]

            Common commands:
              build_ext --inplace    Build extension next to source for local development
              build_ext              Build extension into build/ directory
              install                Install into the active Python environment
              clean --all            Remove temporary build artifacts

            Examples:
              python {script} build_ext --inplace
              python {script} build_ext --inplace --force
              python {script} install

            Notes:
              - Windows uses MSVC + /openmp.
              - Linux/macOS use -fopenmp and require compiler support.
              - Full setuptools help: python {script} --help-setuptools
              - Command help: python {script} build_ext --help
            """
        ).strip()
    )


def main() -> None:
    # Keep command-specific help behavior (e.g., build_ext --help) unchanged.
    if len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"}:
        _print_project_help()
        return

    # Escape hatch for the original setuptools help screen.
    if "--help-setuptools" in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != "--help-setuptools"]
        if len(sys.argv) == 1:
            sys.argv.append("--help")

    setup(
        name="glassbox_sr_core",
        version="0.1",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
    )


if __name__ == "__main__":
    main()

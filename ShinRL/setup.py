from setuptools import find_packages, setup

install_requires = [
    "gym",
    "numpy",
    "opencv-python",
    "matplotlib",
    "seaborn",
    "pandas",
    "pathlib",
    "tqdm",
    "cpprb",
    "torch",
    "structlog",
    "colorama",
    "pyyaml",
    "jupyter"
]

extras_require = {
    "tests": ["pytest", "pytest-benchmark", "pysen[lint]"],
}

setup(
    name="ShinRL",
    version="0.0.1",
    python_requires=">=3.9",
    description=("ShinRL: A python library for analyzing reinforcement learning"),
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)

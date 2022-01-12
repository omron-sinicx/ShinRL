from setuptools import find_packages, setup

install_requires = [
    "gym>=0.20.0",
    "jax>=0.2.21",
    "jaxlib>=0.1.71",
    "tqdm",
    "cpprb",
    "structlog",
    "colorama",
    "pyyaml",
    "seaborn",
    "matplotlib",
    "chex",
    "dm-haiku",
    "optax",
    "distrax>=0.1.0",
    "celluloid",
]

extras_require = {
    "develop": [
        "pytest",
        "pysen[lint]",
    ],
}

setup(
    name="ShinRL-JAX",
    version="0.0.1",
    description="ShinRL: A Library for Evaluating RL Algorithms from Theoretical and Practical Perspectives",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.8",
)

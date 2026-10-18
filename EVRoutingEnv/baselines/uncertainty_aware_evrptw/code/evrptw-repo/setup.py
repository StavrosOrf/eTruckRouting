from setuptools import setup, find_packages

setup(
    name="uncertainty-aware-evrptw",
    version="1.0.0",
    description=(
        "Uncertainty-Aware Deep Reinforcement Learning for Sustainable "
        "Electric Vehicle Routing: A Hybrid Optimization Framework"
    ),
    author="Aymen Jalil Abdulelah",
    author_email="ayman.ja90@uoanbar.edu.iq",
    url="https://github.com/aymenjalil/uncertainty-aware-evrptw",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "tensorboard>=2.13.0",
        "pandas>=2.0.0",
        "networkx>=3.1",
        "tqdm>=4.65.0",
        "click>=8.1.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "milp": ["gurobipy>=10.0.0"],
        "viz":  ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
        "dev":  ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)

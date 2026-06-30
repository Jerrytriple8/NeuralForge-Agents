"""
NeuralForge - AI Agent Orchestration Framework

A comprehensive framework for building, orchestrating, and monitoring
AI agent pipelines with built-in neural network capabilities.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="neuralforge",
    version="2.0.0",
    author="NeuralForge Team",
    author_email="team@neuralforge.ai",
    description="AI Agent Orchestration Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuralforge/neuralforge",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "api": ["fastapi>=0.68.0", "uvicorn>=0.15.0"],
        "cli": ["typer>=0.4.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.15.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "all": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "typer>=0.4.0",
            "pyyaml>=5.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuralforge=cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "neuralforge": ["configs/*.yaml"],
    },
)

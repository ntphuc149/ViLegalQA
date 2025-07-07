#!/usr/bin/env python3
"""
Setup script for ViBidLQA-AQA: Vietnamese Legal Abstractive Question Answering
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
long_description = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"
if REQUIREMENTS_PATH.exists():
    with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f.readlines() 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

# Development dependencies
dev_requirements = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "pre-commit>=3.4.0",
    "jupyter>=1.0.0",
    "ipywidgets>=8.0.0",
]

setup(
    name="vibidlqa-aqa",
    version="1.0.0",
    author="Truong-Phuc Nguyen",
    author_email="ntphuc149@gmail.com",
    description="Vietnamese Legal Abstractive Question Answering System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ntphuc149/ViLegalQA",
    project_urls={
        "Bug Tracker": "https://github.com/ntphuc149/ViLegalQA/issues",
        "Documentation": "https://github.com/ntphuc149/ViLegalQA#readme",
        "Source Code": "https://github.com/ntphuc149/ViLegalQA",
        "Paper": "https://arxiv.org/abs/your-paper-id",
        "Dataset": "https://huggingface.co/datasets/Truong-Phuc/ViBidLQA",
    },
    packages=find_packages(exclude=["tests*", "experiments*", "notebooks*"]),
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "full": requirements + dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "vibidlqa-aqa=scripts.run_aqa:main",
            "vibidlqa-train=scripts.run_aqa:main",
            "vibidlqa-eval=scripts.run_aqa:main",
        ],
    },
    include_package_data=True,
    package_data={
        "configs": ["*.yaml", "**/*.yaml"],
        "scripts": ["*.sh", "**/*.sh"],
    },
    zip_safe=False,
    keywords=[
        "vietnamese", "legal", "question-answering", "nlp", "transformers",
        "fine-tuning", "instruction-tuning", "pytorch", "huggingface"
    ],
)
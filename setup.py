from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="polyclip-t",
    version="0.1.0",
    author="Kelly Larissa VOMO DONFACK et al.",
    author_email="vomodonfack@math.univ-paris13.fr",
    description="Topological Deep Learning for Polygenic Variant Discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MorillaLab/PolyCLIP-T",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "polyclip-t=pipeline.main_pipeline:main",
        ],
    },
)

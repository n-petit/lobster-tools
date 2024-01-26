import setuptools

from lobster_tools import __version__

REQUIRED_PACKAGES = [
    "absl-py",
    "arcticdb",
    "chardet",
    "click",
    "hydra-core",
    "matplotlib",
    "numpy",
    "pandas",
    "requests",
    "scikit-learn",
    "statsmodels",
]
DEV_PACKAGES = [
    "black",
]
CONSOLE_SCRIPTS = [
    "get_sample_data = lobster_tools.data_downloading:get_sample_data",
    "arctic = lobster_tools.arctic_cli:arctic",
    "etf = lobster_tools.arctic_cli:etf",
    "pfmt = lobster_tools.arctic_cli:pfmt",
    "hydra_cli = lobster_tools.experiments:hydra_cli",
]

setuptools.setup(
    name="lobster-tools",
    license="MIT License",
    version=__version__,
    description="Python package for working with LOBSTER data (the limit order book data from Nasdaq).",
    url="https://github.com/n-petit/lobster-tools",
    author="Nicolas Petit",
    author_email="nicolas.petit@keble.ox.ac.uk",
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={"dev": DEV_PACKAGES},
    python_requires=">=3.11",
    include_package_data=True,
    entry_points={
        "console_scripts": CONSOLE_SCRIPTS,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
)

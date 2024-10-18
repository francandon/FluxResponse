from setuptools import setup, find_packages

setup(
    name="FluxResponse",        # Name of your package
    version="0.1",              # Version number
    description="A package for loading and processing response files for M82-axion-decay",  # Short description
    author="Francisco Rodríguez Candón",         # Optional: your name
    packages=find_packages(),   # Automatically find and include all packages
    install_requires=[],        # List any dependencies, e.g., numpy, astropy
    python_requires='>=3.6',    # Specify compatible Python versions
)

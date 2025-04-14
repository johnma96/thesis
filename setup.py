from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="thesis",
    version="0.1.0",
    description="Analytical Master's Thesis",
    package_dir={"":"thesis"},
    author="John Mario Montoya Zapata",
    author_email="jmmontoyaz@unal.edu.co",
    long_description=long_description,
    url="https://github.com/johnma96/thesis.git",
    packages=find_packages(where="thesis"),
    python_requires=">=3.11.3",
    install_requires=install_requires, 
    extras_require={
        "dev": [
                "wheel==0.43.0",
                "notebook==7.2.1"
                ]
    },
)

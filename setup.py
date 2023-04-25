from setuptools import find_packages, setup
from typing import List


def get_requirements(path:str)->List[str]:
    """
    Function to return a list of required packages for this project
    """
    requirements = []

    with open(path, "r") as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if "e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name = "mlops project",
    version = "0.0.2",
    author = "Abomaye Victor",
    author_email = "abomayesan@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)
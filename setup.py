from setuptools import setup, find_packages
from typing import List

def getrequirement()->List[str]:
    requirement_list: List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                
                if requirement and requirement!='-e .':
                    requirement_list.append(requirement)

    except FileNotFoundError:
        print("requirements.txt file not found.")
    return requirement_list

setup(
    name="networksecurity",
    version="0.0.1",
    author="kanish",
    author_email="kanishravesh@gmail.com",
    packages=find_packages(),
    install_requires=getrequirement()
)

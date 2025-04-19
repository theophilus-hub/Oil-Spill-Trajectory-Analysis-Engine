from setuptools import setup, find_packages

setup(
    name="trajectory_core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "requests>=2.25.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.10",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "oil-trajectory=trajectory_core.main:main",
            "oil-trajectory-server=trajectory_core.server:run_server",
        ],
    },
    author="SKAGE.dev",
    author_email="user@example.com",
    description="Oil Spill Trajectory Analysis Engine",
    long_description=open("README_oil_spill.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/oil-spill-trajectory",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

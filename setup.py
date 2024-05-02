from setuptools import setup

setup(
    name="bayes_pde",
    version="0.1.0",
    description="Parallel-in-time probabilistic solutions to PDEs",
    author="Hany Abdulsamad, Sahel Iqbal, Tripp Cator",
    author_email="hany@robot-learning.de",
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "jaxlib",
        "jaxopt",
        "typing_extensions",
        "matplotlib",
        "py-pde",
        "pytest",
    ],
    packages=["bayes_pde"],
    zip_safe=False,
)
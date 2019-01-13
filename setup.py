import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="shrinkage_da",
    version="1.0.0",
    author="Miika Ahdesmaki",
    author_email="miika.ahdesmaki@gmail.com",
    description="Shrinkage Discriminant Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjafin/shrinkage_da/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='shrinkage discriminant analysis',
    install_requires=['scipy>=1.1.0','numpy>=1.15.0'],
    include_package_data=True,
)

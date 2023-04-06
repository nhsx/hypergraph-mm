# Hypergraph Testing
## NHS England -  Digital Analytics and Research Team - PhD Internship Project

### About the Project

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)


[![MkDocs Material](https://img.shields.io/badge/style-MkDocs%20Material-darkblue "Markdown Style: MkDocs")](https://squidfunk.github.io/mkdocs-material/reference/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository holds supporting code for the Transforming Healthcare Data with Graph-based Techniques Using SAIL DataBank project including a copy of the framework built to perform the hypergraph calculations.  A link to the original project proposal can be found [here](https://nhsx.github.io/nhsx-internship-projects/).

_**Note:** Only public or fake data are shared in this repository._

### Project Stucture

The main code is found in the `src` folder of the repository (see Usage below for more information)

```
.
├── docs                    # Documentation
├── notebooks               # Notebooks
├── src                     # Source files
├── .flake8
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENCE
├── mkdocs.yml
├── OPEN_CODE_CHECKLIST.md
├── README.md
└── requirements.txt
```

### Built With

[![Python v3.8](https://img.shields.io/badge/python-v3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
- [Numba](https://numba.pydata.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)

### Getting Started

#### Installation

To get a local copy up and running follow these simple steps.

To clone the repo:

`git clone https://github.com/nhsx/hypergraph-mm`

To create a suitable environment we suggest:
- Build conda environment via `conda create --name hg-test python=3.8`
- Activate environment `conda activate hg-test`
- Install requirements via `python -m pip install -r ./requirements.txt`

The repository uses [pre-commit](https://pre-commit.com) hooks to enforce code style using [Black](https://github.com/psf/black), follows [flake8](https://github.com/PyCQA/flake8), and performs a few other checks.  See `.pre-commit-config.yaml` for more details. These hooks will also need installing locally via:

```{bash}
pre-commit autoupdate
pre-commit install
```

and then will be checked on commit.

### Usage
See the examples in the `notebooks` folder for usage.

### Documentation
The documentation is found in the `docs` folder and uses [mkdocs](https://www.mkdocs.org/).  To make a local copy of the documentation:

`mkdocs serve`

### Roadmap

See the [Issues](https://github.com/nhsx/hypergraph-mm/issues) in GitHub for a list of proposed features (and known issues).

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### License

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

### Contact

To find out more about the [Digital Analytics and Research Team](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch at [england.tdau@nhs.net](mailto:england.tdau@nhs.net).

<!-- ### Acknowledgements -->

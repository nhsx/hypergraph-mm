# Hypergraph Multimorbidity (hypergraph-mm)
## NHS England -  Digital Analytics and Research Team - PhD Internship Projects

### About the Project

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)
![Tests](https://github.com/nhsx/hypergraph-mm/actions/workflows/tests.yml/badge.svg)

[![MkDocs Material](https://img.shields.io/badge/style-MkDocs%20Material-darkblue "Markdown Style: MkDocs")](https://squidfunk.github.io/mkdocs-material/reference/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


`hypergraph-mm` holds supporting code for the "Transforming Healthcare Data with Graph-based Techniques Using SAIL DataBank" project including a copy of the framework built to perform the hypergraph calculations.  It was undertaken over two internships:
- **Jamie Burke** ([GitHub: jaburke166](https://github.com/jaburke166), Wave Three, July-December 2022)
- **Zoe Hancox** ([GitHub: ZoeHancox](https://github.com/ZoeHancox), Wave Four, January-June 2023)

The associated reports from these internships can be found in the [reports](./reports) folder. A link to the original project proposal can be found [here](https://nhsx.github.io/nhsx-internship-projects/transforming-healthcare-data-graph-based-sail-update/) and an associated streamlit app explaining the methodology can be accessed [here](https://nhsx-hypergraphical-streamlit-hypergraphs-hklixt.streamlit.app/) with its GitHub Repository is [hypergraphical](https://github.com/nhsx/hypergraphical).

The repository also supports the work presented in the pre-print [Representing multimorbid disease progressions using directed hypergraphs](https://doi.org/10.1101/2023.08.31.23294903).

_**Note:** Only public or fake data are shared in this repository._

Some of the data used in this project are available in the SAIL Databank at Swansea University, Swansea, UK. All proposals to use SAIL data are subject to review by an independent Information Governance Review Panel (IGRP). Before any data can be accessed, approval must be given by the IGRP. The IGRP carefully considers each project to ensure the proper and appropriate use of SAIL data. When approved, access is gained through a privacy-protecting trusted research environment (TRE) and remote access system referred to as the SAIL Gateway. SAIL has established an application process to be followed by anyone who would like to access data via [SAIL](https://www.saildatabank.com/application-process) - this study has been approved by the IGRP as project 1392.

### Project Stucture

The main code is found in the `hypmm` folder of the repository (see Usage below for more information)

```
.
├── docs                    # Documentation
├── notebooks               # Notebooks
├── hypmm                   # Source files
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

### Roadmap

See the [Issues](https://github.com/nhsx/hypergraph-mm/issues) in GitHub for a list of proposed features (and known issues).

### Testing

Run tests by using `pytest test/test.py` in the top directory.

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### Licence

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENCE](./LICENCE) for more information._

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

### Contact

To find out more about the [Digital Analytics and Research Team](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch at [datascience@nhs.net](mailto:datascience@nhs.net).

### Acknowledgements

This project makes use of anonymised data held in the SAIL Databank, which is part of the national e-health records research infrastructure for Wales. We would like to acknowledge all the data providers who make anonymised data available for research.

<div align="center">
      <h1> Ravnest </h1>
</div>

[![Documentation Status](https://readthedocs.org/projects/ravnest/badge/?version=latest&style=for-the-badge)](http://ravnest.readthedocs.io)

Ravnest introduces a novel asynchronous parallel training approach that combines the best aspects of data and model parallelism. This method enables the distributed training of complex deep learning models across large datasets, utilizing clusters of heterogeneous consumer-grade PCs connected via the internet. Designed with scalability and performance as key objectives, Ravnest seeks to empower researchers and machine learning practitioners. It simplifies the development and deployment of deep learning models, paving the way for innovative research and practical real-world applications.

**Documentation**: https://ravnest.readthedocs.io

**Research Paper**: https://arxiv.org/abs/2401.01728


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

### Installation
```bash
pip install git+https://github.com/ravenprotocol/ravnest.git
```

### Usage

Clone the Repository:
```bash
git clone https://github.com/ravenprotocol/ravnest.git
```

Generate the submodel files:

```bash
python cluster_formation.py
```

> **_NOTE:_**  Uncomment the correct lines in ```cluster_formation.py``` for CNN/ResNet-50/Inception-V3/GPT-Sorter/BERT models.

Execution of Clients (in 3 terminals) for CNN:

Create 3 copies of the ``provider.py`` file inside ``examples/cnn/`` folder. Rename these files as ``provider_0.py``, ``provider_1.py`` and ``provider_2.py``. In each of these files, set the ``name`` parameter of ``Node()`` object to ``'node_0'``, ``'node_1'`` and ``'node_2'``.

```bash
python examples/cnn/provider_0.py
```
```bash
python examples/cnn/provider_1.py
```
```bash
python examples/cnn/provider_2.py
```

> **_NOTE:_** If you have installed Ravnest via Pip, you will have to delete the entire ``ravnest`` subfolder in your cloned directory so that your scripts utilize methods and classes pointing to the pip installed library.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

### Citation
If you have found Ravnest or its foundational components and algorithms to be beneficial in your research, please consider citing the following source:

```
@misc{menon2024ravnest,
      title={Ravnest: Decentralized Asynchronous Training on Heterogeneous Devices}, 
      author={Anirudh Rajiv Menon and Unnikrishnan Menon and Kailash Ahirwar},
      year={2024},
      eprint={2401.01728},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

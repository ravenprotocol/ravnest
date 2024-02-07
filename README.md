# ravnest

Ravnest introduces a novel asynchronous parallel training approach that combines the best aspects of data and model parallelism. This method enables the distributed training of complex deep learning models across large datasets, utilizing clusters of heterogeneous consumer-grade PCs connected via the internet. Designed with scalability and performance as key objectives, Ravnest seeks to empower researchers and machine learning practitioners. It simplifies the development and deployment of deep learning models, paving the way for innovative research and practical real-world applications.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Usage

Compile the protobufs:

```bash
python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. protos/*.proto
```

Generate the submodel files:

```bash
python cluster_formation.py
```

Execution of Clients (in 3 terminals):
```bash
python cnn_client_0.py
```
```bash
python cnn_client_1.py
```
```bash
python cnn_client_2.py
```

<!-- To run infernce on test input:
```bash
python sorter_inference.py
``` -->
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

Research Paper link: https://arxiv.org/abs/2401.01728
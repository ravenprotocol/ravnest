# ravnest
Raven's Nest Architecture

Steps:

Compile the protobufs:

```bash
python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. ./*.proto
```

Generate the submodel files:

```bash
python split_model.py
```

Order of Execution of Clients (in 3 terminals):
```bash
python client_2.py
```
```bash
python client_1.py
```
```bash
python client_0.py
```
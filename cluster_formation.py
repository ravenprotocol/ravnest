import torch
from ravnest.operations.utils import clusterize
# from models import CNN_Net
import random
import numpy as np
from transformers import BertConfig, BertTokenizerFast, BertForPreTraining
# from torchvision.models import resnet50
# from models import inception_v3
# from examples.sorter.mingpt.model_without_padding_mask import GPT
# from examples.sorter.dataset import SortDataset

random.seed(42)
torch.manual_seed(42)
# torch.manual_seed_all(42)
torch.random.manual_seed(42)
np.random.seed(42)

# # For CNN Model  
# model = CNN_Net()

# # For ResNET 50 Model
# model = resnet50(num_classes=200)

# # For Inception V3 Model
# model = inception_v3()

# # For Sorter Model
# train_dataset = SortDataset('train')
# test_dataset = SortDataset('test')
# model_config = GPT.get_default_config()
# model_config.model_type = 'gpt-nano'
# model_config.vocab_size = train_dataset.get_vocab_size()
# model_config.block_size = train_dataset.get_block_size()
# model = GPT(model_config)

# For Bert model
config = BertConfig()
config.return_dict = False

tokenizer = BertTokenizerFast.from_pretrained("bert-base-v2")
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
# input_ids = torch.randint(high=10, size=(32, 10))
model = BertForPreTraining(config)

# clusterize(model=model, example_args=(input_ids,))
clusterize(model=model, example_args=(input_ids,), proportions=[0.7, 0.2, 0.1])
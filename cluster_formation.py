import torch
from torch.utils.data import DataLoader
from ravnest.operations.utils import clusterize
# from models import CNN_Net
import random
import numpy as np
from datasets import load_from_disk
from transformers import BertConfig, BertTokenizerFast, BertForPreTraining, DataCollatorForLanguageModeling
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
# example_args = torch.rand((16,3,64,64))
# clusterize(model=model, example_args=(example_args,), proportions=[0.5, 0.5])

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

tokenizer = BertTokenizerFast.from_pretrained("./examples/bert/data/bert_tokenizer")
tokenized_datasets = load_from_disk('./examples/bert/data/bert_tokenized_wikitext')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=512)
input_s = tokenized_datasets['train']
dataloader = DataLoader(input_s, collate_fn=data_collator, batch_size=8)

test_ip = next(iter(dataloader))#dataloader[0]
input_ids = test_ip['input_ids']
attention_mask = test_ip['attention_mask']
token_type_ids = test_ip['token_type_ids']

# clusterize(model=model, example_args=(input_ids,))
model = BertForPreTraining(config)
clusterize(model=model, example_args=(input_ids,), 
            example_kwargs={'attention_mask':attention_mask, 'token_type_ids':token_type_ids}) #, proportions=[0.5, 0.5])
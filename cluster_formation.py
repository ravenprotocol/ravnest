import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from ravnest import clusterize, set_seed

from models import CNN_Net
from transformers import BertConfig, BertTokenizerFast, BertForPreTraining, DataCollatorForLanguageModeling
from torchvision.models import resnet50
from models import inception_v3
from examples.sorter.mingpt.model_without_padding_mask import GPT
from examples.sorter.dataset import SortDataset

set_seed(42)

# ---------------------- For CNN Model ----------------------

model = CNN_Net()
example_args = torch.rand((64,1,8,8))
clusterize(model=model, example_args=(example_args,))

# ---------------------- For ResNet-50 Model ----------------------

# model = resnet50(num_classes=200)
# example_args = torch.rand((100,3,224,224))
# clusterize(model=model, example_args=(example_args,))

# ---------------------- For Inception-V3 Model ----------------------

# model = inception_v3()
# example_args = torch.rand((64,3,32,32))
# clusterize(model=model, example_args=(example_args,))

# ---------------------- For Sorter Model ----------------------

# First run `python examples/sorter/dataset.py` to generate the pickled dataset.

# train_dataset = SortDataset('train')
# test_dataset = SortDataset('test')
# model_config = GPT.get_default_config()
# model_config.model_type = 'gpt-nano'
# model_config.vocab_size = train_dataset.get_vocab_size()
# model_config.block_size = train_dataset.get_block_size()
# model = GPT(model_config)
# example_args = torch.randint(low=0, high=2, size=(64,11), dtype=torch.int64)
# clusterize(model=model, example_args=(example_args,), pass_data=True)

# ---------------------- For BERT Model ----------------------

# config = BertConfig()
# config.return_dict = False

# tokenizer = BertTokenizerFast.from_pretrained("./examples/bert/data/bert_tokenizer")
# tokenized_datasets = load_from_disk('./examples/bert/data/bert_tokenized_wikitext')

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=512)
# input_s = tokenized_datasets['train']
# dataloader = DataLoader(input_s, collate_fn=data_collator, batch_size=8)

# test_ip = next(iter(dataloader))
# input_ids = test_ip['input_ids']
# attention_mask = test_ip['attention_mask']
# token_type_ids = test_ip['token_type_ids']

# # clusterize(model=model, example_args=(input_ids,))
# model = BertForPreTraining(config)
# clusterize(model=model, example_args=(input_ids,), 
#             example_kwargs={'attention_mask':attention_mask, 'token_type_ids':token_type_ids})

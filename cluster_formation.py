from ravnest.operations.utils import clusterize
from models import CNN_Net
# from torchvision.models import resnet50
# from models import inception_v3
# from examples.sorter.mingpt.model_without_padding_mask import GPT
# from examples.sorter.dataset import SortDataset

# For CNN Model  
model = CNN_Net()

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


clusterize(model=model)
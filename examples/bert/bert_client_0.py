import torch
from datasets import load_from_disk
from ravnest import Node, set_seed
from bert_trainer import BERT_Trainer
from torch.utils.data import DataLoader
from torch_optimizer import Lamb
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling
from transformers.optimization import get_linear_schedule_with_warmup

set_seed(42)

if __name__ == '__main__':

    optimizer=Lamb
    optimizer_params = {'lr':0.00176, 'eps':1e-6, 'debias':True, 'weight_decay':0.01, 'clamp_value':10000.0}
    
    tokenizer = BertTokenizerFast.from_pretrained("./examples/bert/data/bert_tokenizer")
    
    tokenized_datasets = load_from_disk('./examples/bert/data/bert_tokenized_wikitext')

    print(len(tokenized_datasets['train']))
    print(tokenized_datasets['train'][0:2])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=512)

    input_s = tokenized_datasets['train']

    update_frequency = 16
    epochs = 45

    generator = torch.Generator()
    generator.manual_seed(42)
    dataloader = DataLoader(input_s, collate_fn=data_collator, batch_size=8, generator=generator, shuffle=True, drop_last=True)

    if len(dataloader) % update_frequency == 0:
        num_training_steps = len(dataloader)//update_frequency  
    else:
        num_training_steps = len(dataloader)//update_frequency + 1

    num_training_steps *= epochs
    print('Num training steps total: ', num_training_steps)

    scheduler = get_linear_schedule_with_warmup
    scheduler_params = {'num_warmup_steps':5000, 'num_training_steps':num_training_steps}

    node = Node(name = 'node_0', 
                optimizer = optimizer,
                device=torch.device('cuda'),
                update_frequency = update_frequency,
                lr_scheduler = scheduler,
                lr_scheduler_params = scheduler_params,
                lr_step_on_epoch_change=False
                )

    # Custom Trainer Class
    trainer = BERT_Trainer(node=node,
                      train_loader=dataloader,
                      epochs=epochs)

    trainer.train()
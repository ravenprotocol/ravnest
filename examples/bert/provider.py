import torch
from datasets import load_from_disk
from ravnest import Node, set_seed
from bert_trainer import BERT_Trainer
from torch.utils.data import DataLoader
from torch_optimizer import Lamb
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling, BertConfig
from transformers.optimization import get_linear_schedule_with_warmup

set_seed(42)

def preprocess_dataset():
    tokenizer = BertTokenizerFast.from_pretrained("./examples/bert/data/bert_tokenizer")
    
    tokenized_datasets = load_from_disk('./examples/bert/data/bert_tokenized_wikitext')

    print(len(tokenized_datasets['train']))
    print(tokenized_datasets['train'][0:2])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=512)

    input_s = tokenized_datasets['train']

    generator = torch.Generator()
    generator.manual_seed(42)
    train_loader = DataLoader(input_s, collate_fn=data_collator, batch_size=8, generator=generator, shuffle=True, drop_last=True)

    return train_loader

def loss_fn(y_pred, targets):
    labels = targets['labels']
    sentence_order_label = targets['sentence_order_label']

    masked_lm_loss = loss_fct(y_pred[0].view(-1, config.vocab_size), labels.view(-1))
    sentence_order_loss = loss_fct(y_pred[1].view(-1, 2), sentence_order_label.view(-1))
    total_loss = masked_lm_loss + sentence_order_loss
    # opt.zero_grad()
    total_loss.div_(update_frequency)
    return total_loss

if __name__ == '__main__':

    train_loader = preprocess_dataset()

    config = BertConfig()

    optimizer=Lamb
    optimizer_params = {'lr':0.00176, 'eps':1e-6, 'debias':True, 'weight_decay':0.01, 'clamp_value':10000.0}
    loss_fct = torch.nn.CrossEntropyLoss()
    update_frequency = 16
    epochs = 45

    if len(train_loader) % update_frequency == 0:
        num_training_steps = len(train_loader)//update_frequency  
    else:
        num_training_steps = len(train_loader)//update_frequency + 1

    num_training_steps *= epochs
    print('Num training steps total: ', num_training_steps)

    scheduler = get_linear_schedule_with_warmup
    scheduler_params = {'num_warmup_steps':5000, 'num_training_steps':num_training_steps}

    node = Node(name = 'node_0', 
                optimizer = optimizer,
                optimizer_params=optimizer_params,
                update_frequency = update_frequency,              
                lr_scheduler = scheduler,
                lr_scheduler_params = scheduler_params,
                lr_step_on_epoch_change=False,
                criterion = loss_fn, 
                labels = train_loader,
                device=torch.device('cuda')
                )

    # Custom Trainer Class
    trainer = BERT_Trainer(node=node,
                      train_loader=train_loader,
                      epochs=epochs)

    trainer.train()
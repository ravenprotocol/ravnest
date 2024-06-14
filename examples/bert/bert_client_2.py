import torch
import time
from datasets import load_from_disk
from ravnest import Node, set_seed
from torch.utils.data import DataLoader
from torch_optimizer import Lamb
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling, BertConfig
from transformers.optimization import get_linear_schedule_with_warmup

set_seed(42)

if __name__ == '__main__':

    config = BertConfig()

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

    loss_fct = torch.nn.CrossEntropyLoss()

    def loss_fn(y_pred, targets):
        labels = targets['labels']
        sentence_order_label = targets['sentence_order_label']

        masked_lm_loss = loss_fct(y_pred[0].view(-1, config.vocab_size), labels.view(-1))
        sentence_order_loss = loss_fct(y_pred[1].view(-1, 2), sentence_order_label.view(-1))
        total_loss = masked_lm_loss + sentence_order_loss
        # opt.zero_grad()
        total_loss.div_(update_frequency)
        return total_loss

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

    node = Node(name = 'node_2', 
                optimizer = optimizer,
                device=torch.device('cuda'),
                update_frequency = update_frequency,
                lr_scheduler = scheduler,
                lr_scheduler_params = scheduler_params,
                lr_step_on_epoch_change=False,
                criterion = loss_fn, 
                labels = dataloader
                )
    
    while True:
        time.sleep(0)
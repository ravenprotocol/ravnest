import ravnest

class BERT_Trainer(ravnest.Trainer):
    def __init__(self, node=None, train_loader=None, epochs=1):
        super().__init__(node=node, train_loader=train_loader, epochs=epochs)

    def train(self):
        self.prelim_checks()
        for epoch in range(self.epochs):
            for batch in self.train_loader:
                self.node.forward_compute(tensors=batch['input_ids'], 
                                          l_token_type_ids_=batch['token_type_ids'], 
                                          l_attention_mask_=batch['attention_mask'])  

            self.node.wait_for_backwards()   # To be called at end of every epoch
                
        print('BERT Training Done!')

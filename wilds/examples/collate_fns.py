# processing batches of inputs into tensors that can be fed into a model e.g. a BERT model 
import torch


def collate_fn_gen(tokenizer):
    def collate_fn(list_of_examples):
        all_inputs = tokenizer([ex[0] for ex in list_of_examples], max_length=256, padding="max_length", truncation='only_first', return_tensors='pt')
        all_inputs['label'] = torch.tensor([ex[1] for ex in list_of_examples])
        return all_inputs
    return collate_fn

def collate_fn_mlm(list_of_examples):
    pass

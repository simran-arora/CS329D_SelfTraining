# processing batches of inputs into tensors that can be fed into a model e.g. a BERT model 
import random
import torch


def collate_fn_gen(tokenizer):
    def collate_fn(list_of_examples):
        all_inputs = tokenizer([ex[0] for ex in list_of_examples], max_length=256, padding="max_length", truncation='only_first', return_tensors='pt')
        all_inputs['classification_label'] = torch.tensor([ex[1] for ex in list_of_examples])
        return all_inputs
    return collate_fn

def mask(tokenized_inputs, masked_lm_prob, idx2word):
    """
    Nevertheless, consider an example such as:
        Many of the worlds people live with water scarcity, and that percentage will increase as populations increase and climate changes.
    1. First, we tokenize this sentence to get:
        ['many', 'of', 'the', 'worlds', 'people', 'live', 'with', 'water', 'scar', '##city', ',', 'and', 'that', 'percentage', 'will', 'increase', 'as', 'populations', 'increase', 'and', 'climate', 'changes']
    2. Next, we mask out 15% of the tokens (without whole word masking)
        - with 80% probability actually mask it
        - otherwise:
            - keep it the same
            - corrupt it
        TODO: check if we make predictions on 20% of the non-mask tokens
    """
    forbidden_inputs = ["[CLS]", "[SEP]", "[PAD]"]
    sentence_tokenized = tokenized_inputs["input_ids"]
    candidate_indices = []
    labels = []
    for i, token_id in enumerate(sentence_tokenized):
        token = idx2word[token_id]
        labels.append(token_id)
        if token in forbidden_inputs:
            continue
        if len(candidate_indices) >= 1 and len(token) > 2 and token[:2] == "##":
            candidate_indices[-1].append(i)
        else:
            candidate_indices.append([i])
    num_to_predict = max(1, int(len(candidate_indices) * masked_lm_prob))
    chosen_idxs = random.sample(
        list(range(len(candidate_indices))), k=num_to_predict
    )
    chosen_cands = [candidate_indices[idx] for idx in chosen_idxs]
    for index_set in chosen_cands:
        # we are doing whole word masking
        if random.random() < 0.8:
            for idx in index_set:
                sentence_tokenized[idx] = 103
        else:
            if random.random() < 0.5:
                for idx in index_set:
                    masked_token = random.randint(0, len(idx2word) - 1)
                    sentence_tokenized[idx] = masked_token

    # we only want to compute losses over masked / corrupted tokens
    for idx, (label, tok) in enumerate(zip(labels, sentence_tokenized)):
        if label == tok:
            labels[idx] = -100
    # account for [CLS] and [SEP] tokens here
    tokenized_inputs["mlm_label"] = labels
    tokenized_inputs['input_ids'] = sentence_tokenized


def collate_fn_gen_aux_labels(tokenizer, aux_fields):
    idx2word = {idx: word for word, idx in tokenizer.vocab.items()}
    def collate_fn(list_of_examples):
        all_inputs = tokenizer([ex[0] for ex in list_of_examples], max_length=256, padding="max_length", truncation='only_first', return_tensors='pt')
        all_inputs['classification_label'] = torch.tensor([ex[1] for ex in list_of_examples])
        for key, val in aux_fields.items():
            key_idx = int(val[0])
            all_inputs[f'{key}_classification_label'] = torch.tensor([ex[-1][key_idx] for ex in list_of_examples])
        return all_inputs
    return collate_fn

def collate_fn_gen_mlm(tokenizer):
    idx2word = {idx : word for word, idx in tokenizer.vocab.items()}
    def collate_fn(list_of_examples):
        all_tokenized_inputs = []
        for ex in list_of_examples:
            tokenized_input_curr = tokenizer(ex[0], max_length=256, padding="max_length", truncation='only_first')
            mask(tokenized_input_curr, 0.15, idx2word)
            all_tokenized_inputs.append(tokenized_input_curr)
        all_keys = list(all_tokenized_inputs[0].keys())
        out = {key: torch.stack([torch.tensor(ex[key]) for ex in all_tokenized_inputs]) for key in all_keys}
        out['classification_label'] = torch.tensor([ex[1] for ex in list_of_examples])
        return out
    return collate_fn

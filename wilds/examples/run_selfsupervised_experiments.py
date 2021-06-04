import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from datasets import load_from_disk
import torch.nn as nn
import wilds

from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    BertForMaskedLM,
)
from transformers.activations import gelu
from transformers import AutoTokenizer
from transformers import BertModel as Bert, DistilBertModel as DistilBert
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from torch.optim import AdamW, Adam
from collate_fns import *
from self_supervised_trainer import *

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    RandomSampler,
    SequentialSampler,
)

import logging


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ClassificationHead(nn.Module):
    def __init__(self, config, num_labels=3):
        super().__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, output):
        pooled_output = self.pooler(output)
        return self.classifier(self.dropout(pooled_output))

class DistilBertClassificationHead(nn.Module):
    def __init__(self, config, num_labels=3):
        super().__init__()
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def forward(self, hidden_states):
        pooled_output = hidden_states[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        return logits


class DistilBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)
    def forward(self, hidden_states):
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        return prediction_logits


class BERTMultiHead(nn.Module):
    def __init__(self, model_name_or_path, num_labels, mode='classification', aux_output_space=None):
        super().__init__()
        self.num_labels = num_labels
        self.model_type = 'distilbert'
        self.bert = DistilBert.from_pretrained(
            model_name_or_path, return_dict=True
        )
        if aux_output_space:
            self.aux_heads = nn.ModuleDict({f'{key}_classification_label': DistilBertClassificationHead(self.bert.config, num_labels=int(val[1])) for key, val in aux_output_space.items()})
        else:
            self.mlm_head = DistilBertOnlyMLMHead(self.bert.config)
        self.classification_head = DistilBertClassificationHead(self.bert.config, num_labels=self.num_labels)
        self.mode = mode

    def forward(self, input_dicts):
        output = {}
        for input_dict in input_dicts:
            representation = self.get_representation(input_dict)
            classification_labels = [key for key in input_dict if 'classification' in key]
            for key in classification_labels:
                if key == 'classification_label':
                    head_network = self.classification_head
                else:
                    head_network = self.aux_heads[key]
                label_tensor = input_dict[key]
                nkey = key.replace('_label', '')
                if nkey not in output:
                    output[nkey] = self.get_logits(representation, head_network, label_tensor)
                else:
                    curr_logits, curr_labels = output[nkey]
                    new_logits, new_labels = self.get_logits(representation, head_network, label_tensor)
                    output[nkey] = torch.cat([curr_logits, new_logits]),torch.cat([curr_labels, new_labels])
            if 'mlm_label' in input_dict:
                assert('mlm' not in output)
                labels = input_dict['mlm_label']
                output['mlm'] = self.get_logits(representation, self.mlm_head, labels)
        return output

    def get_representation(self, input_dict):
        device = self.bert.device
        input_ids = input_dict["input_ids"].to(device)
        attention_mask = input_dict["attention_mask"].to(device)
        if self.model_type == 'distilbert':
            output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            token_type_ids = input_dict["token_type_ids"].to(device)
            output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        return output.last_hidden_state

    def get_logits(self, representation, classification_head, label_tensor):
        device = self.bert.device
        labels = label_tensor.to(device)
        return classification_head(representation), labels

    def forward_classification(self, input_dict, classification_head, label_tensor):
        device = self.bert.device
        input_ids = input_dict["input_ids"].to(device)
        attention_mask = input_dict["attention_mask"].to(device)
        if self.model_type == 'distilbert':
            output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            token_type_ids = input_dict["token_type_ids"].to(device)
            output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        labels = label_tensor.to(device)
        return classification_head(output.last_hidden_state), labels

    def forward_pretraining(self, input_dict):
        device = self.bert.device
        input_ids = input_dict["input_ids"].to(device)
        attention_mask = input_dict["attention_mask"].to(device)
        if self.model_type == 'distilbert':
            output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            token_type_ids = input_dict["token_type_ids"].to(device)
            output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        logits = self.mlm_head(output.last_hidden_state)
        labels = input_dict['mlm_label'].to(device)
        return logits, labels


def process(ex):
    metadata = ex[2]
    metadata_processed = [val for val in metadata[:-1]] + [-100]
    return ex[0], torch.tensor(-100), torch.tensor(metadata_processed)

@hydra.main(config_path="hydra_config", config_name="aux_self_supervised_bert")
def main(cfg):
    # get labeled data
    orig_working_dir = get_original_cwd()
    log = logging.getLogger(__name__)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    cfg.train.save_dir = os.path.join(orig_working_dir, cfg.train.save_dir)
    if cfg.mode == 'eval':
        dataset_version = f'shikhar-test/labeled_{cfg.dataset}.csv'
    else:
        dataset_version=f'shikhar-test/subsample_{cfg.dataset}.csv'
    full_dataset = wilds.get_dataset(
        dataset=cfg.dataset,
        version=None,
        root_dir='{}/{}'.format(orig_working_dir,cfg.root_dir),
        download=False,
        split_scheme='official',
        dataset_version=dataset_version,
        )
    if 'aux' in cfg.mode:
        aux_labels = {key: eval(val) for key, val in cfg['aux_labels'].items()}
        model = BERTMultiHead(cfg.model_name_or_path, full_dataset.n_classes, aux_output_space=aux_labels)
    else:
        model = BERTMultiHead(cfg.model_name_or_path, full_dataset.n_classes, aux_output_space=None)
    model.to(device)
    if 'distil' in cfg.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    load_path = cfg.get('load_path', None)
    if load_path:
        absolute_path = os.path.join(orig_working_dir, load_path)
        log.info(f"Loading from: {absolute_path}")
        model.load_state_dict(torch.load(absolute_path))
    trainer = Trainer(cfg.train, log)
    if cfg.mode == 'train_labeled':
        collate_fn = collate_fn_gen(tokenizer)
        split_dict = full_dataset.split_dict
        train = full_dataset.get_subset('train')
        if 'id_val' in split_dict:
            val = full_dataset.get_subset('id_val')
        else:
            val = full_dataset.get_subset('val')
        train_data_task = DataLoader(
            train, sampler=RandomSampler(train), batch_size=cfg.train.batch_size,
            collate_fn=collate_fn
        )
        eval_data_task = DataLoader(
            val, sampler=SequentialSampler(val), batch_size=cfg.train.eval_batch_size,
            collate_fn=collate_fn
        )
        train_data_func = {'classification': train_data_task}
        eval_data_func = {'classification': eval_data_task}
        trainer.train(model, train_data_func, eval_data_func)
    elif cfg.mode == 'train_aux':
        collate_fn = collate_fn_gen_aux_labels(tokenizer, aux_labels)
        all_examples = full_dataset.get_subset('test')
        num_examples = len(all_examples)
        train_idx = int(0.8 * num_examples)
        # we set the labels to -100, because we do not want to use them.
        train_ss = [process(all_examples[idx]) for idx in range(0, train_idx)]
        val_ss = [process(all_examples[idx]) for idx in range(train_idx, num_examples)]
        random.shuffle(val_ss)
        train_data_task = DataLoader(
            train_ss, sampler=RandomSampler(train_ss), batch_size=cfg.train.batch_size,
            collate_fn=collate_fn
        )
        eval_data_task = DataLoader(
            val_ss, sampler=SequentialSampler(val_ss), batch_size=cfg.train.eval_batch_size,
            collate_fn=collate_fn
        )
        train_data_func = {'classification': train_data_task}
        eval_data_func = {'classification': eval_data_task}
        trainer.train(model, train_data_func, eval_data_func)
    elif cfg.mode == 'train_mtl_aux':
        collate_fn = collate_fn_gen_aux_labels(tokenizer, aux_labels)
        split_dict = full_dataset.split_dict
        train = full_dataset.get_subset('train')
        first_k = cfg.get('first_k', -1)
        if first_k != -1:
            log.info(f"Training on {first_k} examples")
            train = [train[idx] for idx in range(0, first_k)]
        if 'id_val' in split_dict:
            val = full_dataset.get_subset('id_val')
        else:
            val = full_dataset.get_subset('val')
        use_target_domain = cfg.get('use_target_domain', False)
        if use_target_domain:
            all_examples = full_dataset.get_subset('test')
            num_examples = len(all_examples)
            train_idx = int(0.8 * num_examples)
            # we set the labels to -100, because we do not want to use them.
            train_test_domain_ss = [process(all_examples[idx]) for idx in range(0, train_idx)]
            val_test_domain_ss = [process(all_examples[idx]) for idx in range(train_idx, num_examples)]
            train_ss = train_test_domain_ss + [ex for ex in train]
            val_ss = val_test_domain_ss + [ex for ex in val]
            random.shuffle(val_ss)
        else:
            train_ss = train
            val_ss = val
        train_data_task = DataLoader(
            train_ss, sampler=RandomSampler(train_ss), batch_size=cfg.train.batch_size,
            collate_fn=collate_fn
        )
        eval_data_task = DataLoader(
            val_ss, sampler=SequentialSampler(val_ss), batch_size=cfg.train.eval_batch_size,
            collate_fn=collate_fn
        )
        train_data_func = {'classification': train_data_task}
        eval_data_func = {'classification': eval_data_task}
        trainer.train(model, train_data_func, eval_data_func)
    elif cfg.mode == 'train_mtl':
        model.mode = 'train_mtl'
        # TODO: add a flag such that we make a prediction even on the MLMed examples, if we do have a label present. 
        # TODO: to do this, just need to feed this flag to the collate_fn_gen function
        collate_fn = collate_fn_gen(tokenizer)
        collate_fn_mlm = collate_fn_gen_mlm(tokenizer)
        split_dict = full_dataset.split_dict
        train = full_dataset.get_subset('train')
        first_k = cfg.get('first_k', -1)
        if first_k != -1:
            log.info(f"Training on {first_k} examples")
            train = [train[idx] for idx in range(0, first_k)]
        if 'id_val' in split_dict:
            val = full_dataset.get_subset('id_val')
        else:
            val = full_dataset.get_subset('val')
        use_target_domain = cfg.get('use_target_domain', False)
        if use_target_domain:
            all_examples = full_dataset.get_subset('test')
            num_examples = len(all_examples)
            train_idx = int(0.8 * num_examples)
            # we set the labels to -100, because we do not want to use them.
            train_test_domain_mlm = [process(all_examples[idx]) for idx in range(0, train_idx)]
            val_test_domain_mlm = [process(all_examples[idx]) for idx in range(train_idx, num_examples)]
            train_mlm = train_test_domain_mlm + [ex for ex in train]
            val_mlm = val_test_domain_mlm + [ex for ex in val]
            random.shuffle(val_mlm)
        else:
            train_mlm = train
            val_mlm = val

        train_data_task = DataLoader(
            train, sampler=RandomSampler(train), batch_size=cfg.train.batch_size,
            collate_fn=collate_fn
        )
        eval_data_task = DataLoader(
            val, sampler=SequentialSampler(val), batch_size=cfg.train.eval_batch_size,
            collate_fn=collate_fn
        )
        train_data_mlm = DataLoader(
            train_mlm, sampler=RandomSampler(train_mlm), batch_size=cfg.train.batch_size,
            collate_fn=collate_fn_mlm
        )
        eval_data_mlm = DataLoader(
            val_mlm, sampler=SequentialSampler(val_mlm), batch_size=cfg.train.eval_batch_size,
            collate_fn=collate_fn_mlm
        )
        train_data_func = {'classification': train_data_task, 'mlm': train_data_mlm}
        eval_data_func = {'classification': eval_data_task, 'mlm': eval_data_mlm}
        trainer.train(model, train_data_func, eval_data_func)
    elif cfg.mode == 'train_unlabeled':
        # we finetune via the self-supervised loss on unlabeled examples here.
        collate_fn_mlm = collate_fn_gen_mlm(tokenizer)
        all_examples = full_dataset.get_subset('test')
        num_examples = len(all_examples)
        train_idx = int(0.8 * num_examples)
        train = [process(all_examples[idx]) for idx in range(0, train_idx)]
        val = [process(all_examples[idx]) for idx in range(train_idx, num_examples)]
        train_data_mlm = DataLoader(
            train, sampler=RandomSampler(train), batch_size=cfg.train.batch_size,
            collate_fn=collate_fn_mlm)
        eval_data_mlm = DataLoader(
            val, sampler=SequentialSampler(val), batch_size=cfg.train.eval_batch_size,
            collate_fn=collate_fn_mlm)
        #TODO: covert to data loaders
        train_data_func = {'mlm': train_data_mlm}
        eval_data_func = {'mlm': eval_data_mlm}
        trainer.train(model, train_data_func, eval_data_func)
    elif cfg.mode in ['eval', 'eval_aux']:
        collate_fn = collate_fn_gen(tokenizer)
        split_dict = full_dataset.split_dict
        use_val_iid = cfg.get('use_val_iid', False)
        if use_val_iid:
            if 'id_val' in split_dict:
                test = full_dataset.get_subset('id_val')
            else:
                test = full_dataset.get_subset('val')
        else:
            test = full_dataset.get_subset('test')
        test_time_train = cfg.get('test_time_train', False)
        if test_time_train:
            collate_fn_mlm = collate_fn_gen_mlm(tokenizer)
            eval_data_task = DataLoader(
                test, sampler=SequentialSampler(test), batch_size=1,
                collate_fn=collate_fn)
            result, logits = trainer.eval_test_time_train(model, test, collate_fn_mlm, collate_fn)
            torch.save(logits, 'output_logits.obj')
        else:
            eval_data_task = DataLoader(
                test, sampler=SequentialSampler(test), batch_size=cfg.train.eval_batch_size,
                collate_fn=collate_fn)
            result = trainer.eval(model, {'classification': eval_data_task})
        print(result)

if __name__ == '__main__':
    main()

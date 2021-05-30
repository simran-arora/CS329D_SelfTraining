from torch.optim import AdamW, Adam
from utils import *
from datasets import Dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

import copy
import os
import torch
import torch.nn as nn


from hydra import utils
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_constant_schedule



class Trainer():
    def __init__(self, args, log):
        self.args = args
        self.log = log

    def eval(self, model, eval_func):
        total_ex = sum([len(val.dataset) for _, val in eval_func.items()])
        batches = zip(*[dataloader for _, dataloader in eval_func.items()])
        metrics = {key: (0.0, 0.0, 0.0) for key in eval_func}
        def update(logits, labels, key):
            if key == "mlm":
                vocab_size = logits.shape[-1]
                logits = logits.reshape(-1, vocab_size)
                labels = labels.reshape(-1)
            val_acc, val_loss, total = metrics[key]
            val_acc += torch.sum((torch.argmax(logits, dim=1) == labels).float()).item()
            val_loss += loss_fn(logits, labels).detach().item()
            total += len(labels)
            metrics[key] = val_acc, val_loss, total
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        with torch.no_grad(), tqdm(total=total_ex) as progress_bar:
            for b_idx, batch in enumerate(batches):
                out_dict = model(batch)
                early_stop_idx = self.args.get("max_eval_batches", 100000)
                if b_idx >= early_stop_idx:
                    break
                for key in out_dict:
                    logits, labels = out_dict[key]
                    update(logits, labels, key)
                    progress_bar.update(len(logits))
        for key in metrics:
            acc, loss, total = metrics[key]
            metrics[key] = acc / total, loss / total
        return metrics

    def get_batches(self, train_data_func):
        if type(train_data_func) == dict:
            batches = zip(
                *[dataloader for _, dataloader in train_data_func.items()]
            )
        else:
            batches = train_data_func
        return batches

    def get_total(self, train_data_func):
        accum_steps = self.args.get('accum_steps', 1)
        if type(train_data_func) == dict:
            total = min([len(dataloader.dataset) for _, dataloader in train_data_func.items()])
        else:
            total = len(train_data_func.dataset)
            num_losses = 1
        effective_batch_size = self.args.batch_size * accum_steps
        num_steps_per_epoch = 1 + (total // effective_batch_size)
        return self.args.num_epochs * num_steps_per_epoch, num_steps_per_epoch


    def get_opt(self, model):
        no_decay = ["bias", "LayerNorm.weight"]
        weight_decay = self.args.get('weight_decay', 0.0)
        adam_epsilon = self.args.get('adam_epsilon', 1e-7)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.lr,
            eps=adam_epsilon,
        )
        return optimizer

    def forward_pass(self, model, batch, progress_bar, accum_steps, loss_fn):
        model.train()
        out = model(batch)
        losses_for_tf = {}
        assert type(out) == dict
        out_dict = out
        loss = 0.0
        for key in out_dict:
            logits, labels = out_dict[key]
            progress_bar.update(len(logits))
            if key == "mlm":
                vocab_size = logits.shape[-1]
                logits = logits.reshape(-1, vocab_size)
                labels = labels.reshape(-1)
            curr_loss = loss_fn(logits, labels)
            losses_for_tf[key] = curr_loss.item()
            loss += curr_loss
        loss /= accum_steps
        loss.backward()
        return loss.item(), losses_for_tf

    def get_scheduler(self, opt, t_total):
        scheduler = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )
        return scheduler


    def train(self, model, train_data_func, eval_func):
        opt = self.get_opt(model)
        accum_steps = self.args.get('accum_steps', 1)
        t_total, epoch_size = self.get_total(train_data_func)
        # epoch_size is the number of updates within an epoch 
        # t_total is the total number of updates we are going to perform overall
        if self.args.get('warmup_steps', -1) >= 0:
            scheduler = self.get_scheduler(opt, t_total)
        else:
            scheduler = get_constant_schedule(opt)
        num_epochs = self.args.num_epochs
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        num_steps = 0
        best_acc = -1.0
        tbx = SummaryWriter(self.args.save_dir)
        for epoch_num in tqdm(range(num_epochs)):
            self.log.info(f"Epoch: {epoch_num}")
            losses = []
            # epoch_size = k * dataset_size and the effective batch size is self.args.batch_size * accum_steps
            # we only evaluate once per epoch but we can do epoch_size // k if we want to perform multiple evaluations per epoch 
            EVAL_EVERY = epoch_size
            print("evaluating every {} iterations".format(EVAL_EVERY))
            batches = self.get_batches(train_data_func)
            with torch.enable_grad(), tqdm(total=epoch_size) as progress_bar:
                loss_curr = 0.0
                losses_for_tf_all = []
                model.zero_grad()
                for batch in batches:
                    loss_curr_iter, losses_for_tf_dict = self.forward_pass(model, batch, progress_bar, accum_steps, loss_fn)
                    losses_for_tf_all.append(losses_for_tf_dict)
                    loss_curr += loss_curr_iter
                    if len(losses_for_tf_all) == accum_steps:
                        opt.step()
                        scheduler.step()
                        model.zero_grad()
                        losses_for_tf_agg = {
                            key: np.sum([d[key] for d in losses_for_tf_all])
                            for key in losses_for_tf_all[0]
                        }
                        progress_bar.set_postfix(epoch=epoch_num, NLL=loss_curr)
                        for key in losses_for_tf_agg:
                            tbx.add_scalar(
                                f"train/{key}", losses_for_tf_agg[key], num_steps
                            )
                        tbx.add_scalar("train/NLL", loss_curr, num_steps)
                        # evaluations should only happen when gradients are not stored. 
                        if num_steps % EVAL_EVERY == 0:
                            model.eval()
                            self.log.info(f"Evaluating at step {num_steps}...")
                            eval_dict = self.eval(model, eval_func)
                            for key in eval_dict:
                                dict_curr = {
                                    "acc": eval_dict[key][0],
                                    "loss": eval_dict[key][1],
                                }
                                results_str = ", ".join(
                                    f"{k}: {v:05.2f}" for k, v in dict_curr.items()
                                )
                                self.log.info(f"Eval {key} {results_str}")
                                for k, v in dict_curr.items():
                                    tbx.add_scalar(f"val/{key}_{k}", v, num_steps)
                            curr_acc = eval_dict["classification"][0]
                            if curr_acc > best_acc:
                                best_acc = curr_acc
                                self.log.info(f"Saving model to {self.args.save_path}")
                                torch.save(model.state_dict(), self.args.save_path)
                        num_steps += 1
                        loss_curr = 0.0
                        losses_for_tf_all = []
        return
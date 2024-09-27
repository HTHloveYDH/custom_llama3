import os
import time
from collections import OrderedDict

import torch
import torch.distributed as dist

from utils.get_device_type import get_device_type


def st_train_on_epoch(model, raw_model, data_loader, optimizer, device:str, steps_per_epoch:int, \
                      grad_accum_steps:int, epoch:int, log_interval:int, dp:bool, tp:bool, \
                      master_process:bool):
    model.train()
    loss_tracker = []
    device_type = get_device_type(device)
    for step in range(steps_per_epoch):
        start_time = time.time()
        loss_accum = 0.0
        # x, y = data_loader.get_batch_data()  # only for demo
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if dp or tp:
            model.require_backward_grad_sync = ((step + 1) % grad_accum_steps == 0)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = model(x)
            loss = raw_model.compute_loss(logits, y, tp)
        loss_accum = loss.detach()
        loss.backward()
        if dp or tp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # all_reduce (mean)
        # update weights at the 'grad_accum_steps'st step of every 'grad_accum_steps' steps
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            # clear history gradients before the next gradients accumulation period starts
            optimizer.zero_grad()
        if device_type == 'cuda':
            torch.cuda.synchronize() # wait for the GPU to finish work
        batch_time = time.time() - start_time
        # print log and save training history
        if master_process and step % log_interval == 0:
            loss_tracker.append(loss_accum.item())  # .item(): to scalar
            print(f'[train] cost {batch_time}s for one batch')

@torch.no_grad()
def st_valid_on_epoch(model, raw_model, data_loader, device:str, val_steps:int, epoch:int, \
                      dp:bool, tp:bool, master_process:bool, lora:bool):
    model.eval()
    val_loss_tracker = []
    device_type = get_device_type(device)
    val_loss_accum = 0.0
    for _ in range(val_steps):
        # x, y = data_loader.get_batch_data()  # only for demo
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = model(x)
            loss = raw_model.compute_loss(logits, y, tp)
        loss = loss / val_steps
        val_loss_accum += loss.detach()
    if dp or tp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)  # all_reduce (mean)
        val_loss_tracker.append(val_loss_accum.item())
    if master_process:
        print(f'validation loss: {val_loss_accum.item():.4f}')
        # optionally write model checkpoints
        log_dir = os.path.join('.', 'log', 'ckpt')
        save_curr_model_path = os.path.join(log_dir, f'model_epoch_{epoch}.pt')
        _save_ckpt(raw_model, epoch, val_loss_accum.item(), save_curr_model_path, lora)
        checkpoint_path = os.path.join('.', 'ckpt', f'model.pt')
        _save_ckpt(raw_model, epoch, val_loss_accum.item(), checkpoint_path, lora)

def dpo_train_on_epoch(model, raw_model, data_loader, optimizer, device:str, steps_per_epoch:int, \
                       grad_accum_steps:int, epoch:int, log_interval:int, dp:bool, tp:bool, \
                       master_process:bool):
    model.train()
    loss_tracker = []
    device_type = get_device_type(device)
    for step in range(steps_per_epoch):
        start_time = time.time()
        loss_accum = 0.0
        x_winner, x_loser = data_loader.next_batch()
        x_winner, x_loser = x_winner.to(device), x_loser.to(device)
        if dp or tp:
            model.require_backward_grad_sync = ((step + 1) % grad_accum_steps == 0)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            winner_values, winner_logits = model(x_winner)
            loser_values, loser_logits = model(x_loser)
            # compute dpo loss
            # loss = dpo_loss(winner_values.mean(dim=-1), loser_values.mean(dim=-1))
            loss = raw_model.dpo_loss(winner_values[:, -1], loser_values[:, -1], tp)
        loss_accum = loss.detach()
        loss.backward()
        if dp or tp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # all_reduce (mean)
        # update weights at the 'grad_accum_steps'st step of every 'grad_accum_steps' steps
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            # clear history gradients before the next gradients accumulation period starts
            optimizer.zero_grad()
        if device_type == 'cuda':
            torch.cuda.synchronize() # wait for the GPU to finish work
        batch_time = time.time() - start_time
        # print log and save training history
        if master_process and step % log_interval == 0:
            loss_tracker.append(loss_accum.item())  # .item(): to scalar
            print(f'[train] cost {batch_time}s for one batch')

@torch.no_grad()
def dpo_valid_on_epoch(model, raw_model, data_loader, device:str, val_steps:int, epoch:int, \
                       dp:bool, tp:bool, master_process:bool, lora:bool):
    model.eval()
    val_loss_tracker = []
    device_type = get_device_type(device)
    val_loss_accum = 0.0
    for _ in range(val_steps):
        x_winner, x_loser = data_loader.next_batch()
        x_winner, x_loser = x_winner.to(device), x_loser.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            winner_values, winner_logits = model(x_winner)
            loser_values, loser_logits = model(x_loser)
            # compute dpo loss
            # loss = dpo_loss(winner_values.mean(dim=-1), loser_values.mean(dim=-1))
            loss = raw_model.dpo_loss(winner_values[:, -1], loser_values[:, -1], tp)
        loss = loss / val_steps
        val_loss_accum += loss.detach()
    if dp or tp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)  # all_reduce (mean)
        val_loss_tracker.append(val_loss_accum.item())
    if master_process:
        print(f'validation loss: {val_loss_accum.item():.4f}')
        # optionally write model checkpoints
        log_dir = os.path.join('.', 'log', 'ckpt')
        save_curr_model_path = os.path.join(log_dir, f'model_epoch_{epoch}.pt')
        _save_ckpt(raw_model, epoch, val_loss_accum.item(), save_curr_model_path, lora)
        checkpoint_path = os.path.join('.', 'ckpt', f'model.pt')
        _save_ckpt(raw_model, epoch, val_loss_accum.item(), checkpoint_path, lora)

def get_optimizer(raw_model, weight_decay:float, learning_rate:float):
    return torch.optim.Adam(raw_model.parameters())

def _save_full_ckpt(model, epoch:int, val_loss_accum:float, checkpoint_path:str):
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch,
        'params': model.params,
        'val_loss': val_loss_accum
    }
    # you might also want to add optimizer.state_dict() and
    # rng seeds etc., if you wanted to more exactly resume training
    torch.save(checkpoint, checkpoint_path)

def _save_lora_ckpt(model, epoch:int, val_loss_accum:float, checkpoint_path:str):
    checkpoint = {
        'model': OrderedDict(),
        'epoch': epoch,
        'params': model.params,
        'val_loss': val_loss_accum
    }
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        if 'lora' in key:
            checkpoint['model'].update({key: model_state_dict[key]})
    torch.save(checkpoint, './lora.pt')

def _save_ckpt(model, epoch:int, val_loss_accum:float, checkpoint_path:str, lora:bool):
    if lora:
        _save_lora_ckpt(model, epoch, val_loss_accum, checkpoint_path)
    else:
        _save_full_ckpt(model, epoch, val_loss_accum, checkpoint_path)

def resume_from_ckpt(model, ckpt_dir:str):
    checkpoint_path = os.path.join(ckpt_dir, 'model.pt')
    if os.path.exists(checkpoint_path):
        print('Loading checkpoint directory')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
    return model

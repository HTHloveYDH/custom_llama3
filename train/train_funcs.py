import os
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.distributed.tensor.parallel import loss_parallel

from dist.ParallelArgs import ParallelArgs
from utils.get_device_type import get_device_type


def st_forward_pass(model, x, y, parallel_args:ParallelArgs):
    logits = model(x)
    if parallel_args.dp > 1:
        loss = model.module.compute_loss(logits, y, parallel_args.tp > 1, parallel_args.parallel_loss)
    else:
        loss = model.compute_loss(logits, y, parallel_args.tp > 1, parallel_args.parallel_loss)
    return loss

def st_backward_pass(loss, parallel_args:ParallelArgs):
    if parallel_args.parallel_loss:
        with loss_parallel():
            loss.backward()
    else:
        loss.backward()

def pp_st_forward_pass(model, x, y, parallel_args:ParallelArgs):
    logits = model(x)
    if parallel_args.dp > 1:
        loss = model.module.compute_loss(logits, y, parallel_args.tp > 1, parallel_args.parallel_loss)
    else:
        loss = model.compute_loss(logits, y, parallel_args.tp > 1, parallel_args.parallel_loss)
    return loss

def pp_st_backward_pass(loss, parallel_args:ParallelArgs):
    if parallel_args.parallel_loss:
        with loss_parallel():
            loss.backward()
    else:
        loss.backward()

def st_train_on_epoch(model, data_loader, optimizer, device:str, steps_per_epoch:int, \
                      grad_accum_steps:int, epoch:int, log_interval:int, \
                      parallel_args:ParallelArgs, master_process:bool, pp_schedule):
    model.train()
    loss_tracker = []
    device_type = get_device_type(device)
    parallel = parallel_args.dp > 1 or parallel_args.tp > 1 or parallel_args.pp > 1
    for step in range(steps_per_epoch):
        start_time = time.time()
        loss_accum = 0.0
        # x, y = data_loader.get_batch_data()  # only for demo
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if parallel:
            model.require_backward_grad_sync = ((step + 1) % grad_accum_steps == 0)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            if parallel_args.pp > 1:
                loss = pp_st_forward_pass(model, x, y, parallel_args)
            else:
                loss = st_forward_pass(model, x, y, parallel_args)
        loss_accum = loss.detach()
        # backward
        if parallel_args.pp > 1:
            pp_st_backward_pass(loss, parallel_args)
        else:
            st_backward_pass(loss, parallel_args)
        if parallel and not parallel_args.parallel_loss:
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
def st_valid_on_epoch(model, data_loader, device:str, val_steps:int, \
                      epoch:int, parallel_args:ParallelArgs, \
                      master_process:bool, lora:bool, pp_schedule):
    model.eval()
    val_loss_tracker = []
    device_type = get_device_type(device)
    parallel = parallel_args.dp > 1 or parallel_args.tp > 1 or parallel_args.pp > 1
    val_loss_accum = 0.0
    for _ in range(val_steps):
        # x, y = data_loader.get_batch_data()  # only for demo
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            if parallel_args.pp > 1:
                loss = st_forward_pass(model, x, y, parallel_args)
            else:
                loss = pp_st_forward_pass(model, x, y, parallel_args)
        loss = loss / val_steps
        val_loss_accum += loss.detach()
    if parallel and not parallel_args.parallel_loss:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)  # all_reduce (mean)
        val_loss_tracker.append(val_loss_accum.item())
    if master_process:
        print(f'validation loss: {val_loss_accum.item():.4f}')
        # optionally write model checkpoints
        log_dir = os.path.join('.', 'log', 'ckpt')
        save_curr_model_path = os.path.join(log_dir, f'model_epoch_{epoch}.pt')
        checkpoint_path = os.path.join('.', 'ckpt', f'model.pt')
        if parallel_args.dp > 1:
            _save_ckpt(model.module, epoch, val_loss_accum.item(), save_curr_model_path, lora)
            _save_ckpt(model.module, epoch, val_loss_accum.item(), checkpoint_path, lora)
        else:
            _save_ckpt(model, epoch, val_loss_accum.item(), save_curr_model_path, lora)
            _save_ckpt(model, epoch, val_loss_accum.item(), checkpoint_path, lora)

def dpo_forward_pass(model, x_winner, x_loser, parallel_args:ParallelArgs):
    winner_values, winner_logits = model(x_winner)
    loser_values, loser_logits = model(x_loser)
    # compute dpo loss
    if parallel_args.dp > 1:
        # loss = model.module.dpo_loss(winner_values.mean(dim=-1), loser_values.mean(dim=-1))
        loss = model.module.dpo_loss(
            winner_values[:, -1], loser_values[:, -1], parallel_args.tp > 1
        )
    else:
        # loss = model.dpo_loss(winner_values.mean(dim=-1), loser_values.mean(dim=-1))
        loss = model.dpo_loss(
            winner_values[:, -1], loser_values[:, -1], parallel_args.tp > 1
        )
    return loss

def dpo_backward_pass(loss, parallel_args:ParallelArgs):
    if parallel_args.parallel_loss:
        with loss_parallel():
            loss.backward()
    else:
        loss.backward()

def dpo_train_on_epoch(model, data_loader, optimizer, device:str, steps_per_epoch:int, \
                       grad_accum_steps:int, epoch:int, log_interval:int, \
                       parallel_args:ParallelArgs, master_process:bool, pp_schedule):
    # TODO: add pipeline parallel for DPO
    assert parallel_args.pp == 1
    model.train()
    loss_tracker = []
    device_type = get_device_type(device)
    parallel = parallel_args.dp > 1 or parallel_args.tp > 1
    for step in range(steps_per_epoch):
        start_time = time.time()
        loss_accum = 0.0
        x_winner, x_loser = data_loader.next_batch()
        x_winner, x_loser = x_winner.to(device), x_loser.to(device)
        if parallel:
            model.require_backward_grad_sync = ((step + 1) % grad_accum_steps == 0)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            loss = dpo_forward_pass(model, x_winner, x_loser, parallel_args)
        loss_accum = loss.detach()
        # backward
        dpo_backward_pass(loss, parallel_args)
        if parallel and not parallel_args.parallel_loss:
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
def dpo_valid_on_epoch(model, data_loader, device:str, val_steps:int, \
                       epoch:int, parallel_args:ParallelArgs, \
                       master_process:bool, lora:bool, pp_schedule):
    # TODO: add pipeline parallel for DPO
    assert parallel_args.pp == 1
    model.eval()
    val_loss_tracker = []
    device_type = get_device_type(device)
    parallel = parallel_args.dp > 1 or parallel_args.tp > 1
    val_loss_accum = 0.0
    for _ in range(val_steps):
        x_winner, x_loser = data_loader.next_batch()
        x_winner, x_loser = x_winner.to(device), x_loser.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            winner_values, winner_logits = model(x_winner)
            loser_values, loser_logits = model(x_loser)
            # compute dpo loss
            if parallel_args.dp > 1:
                # loss = model.module.dpo_loss(winner_values.mean(dim=-1), loser_values.mean(dim=-1))
                loss = model.module.dpo_loss(
                    winner_values[:, -1], loser_values[:, -1], parallel_args.tp > 1
                )
            else:
                # loss = model.dpo_loss(winner_values.mean(dim=-1), loser_values.mean(dim=-1))
                loss = model.dpo_loss(
                    winner_values[:, -1], loser_values[:, -1], parallel_args.tp > 1
                )
        loss = loss / val_steps
        val_loss_accum += loss.detach()
    if parallel and not parallel_args.parallel_loss:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)  # all_reduce (mean)
        val_loss_tracker.append(val_loss_accum.item())
    if master_process:
        print(f'validation loss: {val_loss_accum.item():.4f}')
        # optionally write model checkpoints
        log_dir = os.path.join('.', 'log', 'ckpt')
        save_curr_model_path = os.path.join(log_dir, f'model_epoch_{epoch}.pt')
        checkpoint_path = os.path.join('.', 'ckpt', f'model.pt')
        if parallel_args.dp > 1:
            _save_ckpt(model.module, epoch, val_loss_accum.item(), save_curr_model_path, lora)
            _save_ckpt(model.module, epoch, val_loss_accum.item(), checkpoint_path, lora)
        else:
            _save_ckpt(model, epoch, val_loss_accum.item(), save_curr_model_path, lora)
            _save_ckpt(model, epoch, val_loss_accum.item(), checkpoint_path, lora)

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

def _clip_norm(model, max_norm:float=1.0):
    if isinstance(model, list):
        modules = model
    elif isinstance(model, nn.Module):
        modules = [model]
    else:
        modules = [model.module]
    # clip gradients
    for module in modules:
        torch.nn.utils.clip_grad_norm_(
            module.parameters(), max_norm, foreach=True
        )

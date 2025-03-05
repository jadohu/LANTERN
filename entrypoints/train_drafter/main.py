import os
import json
import argparse

import torch
import wandb

from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors import safe_open
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from models.configs.configs import EConfig

from .data_utils import (
    list_files,
    AddGaussianNoise,
    AddUniformNoise,
    CustomDataset,
    CoupledDataset,
    DataCollatorWithPadding,
    DataCollatorWithPaddingForCoupled,
)

torch.backends.cuda.matmul.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser(description='Training drafter')
    
    # paths and directories
    parser.add_argument("--model", type=str, default="lumina_mgpt")
    parser.add_argument('--base_path', type=str, default='ckpts/lumina_mgpt/Lumina-mGPT-7B-768')
    parser.add_argument('--config_path', type=str, default='data/configs/lumina_mgpt_config.json')
    # parser.add_argument('--data_dir', type=str, default='data/drafter_train_data/lumina_mgpt/mscoco2017train')
    parser.add_argument('--data_dir', type=str, default='/home/server44/sihwan_workspace/ssd/lumina_mgpt_eagle_mscoco2017train')
    parser.add_argument('--save_dir', type=str, default='ckpts/lumina_mgpt/trained_drafters')
    
    # dataset arguments
    parser.add_argument('--coupled', action='store_true', default=False)
    parser.add_argument('--train_data_ratio', type=float, default=0.95)
    parser.add_argument('--data_noise', type=str, default='uniform')
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--std', type=float, default=0.2)
    
    # training arguments
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--warmup_steps_ratio', type=float, default=0.03)
    parser.add_argument('--is_warmup', action='store_true', default=True)
    
    parser.add_argument('--p_w', type=float, default=0.1)
    parser.add_argument('--cfg_loss', action='store_true', default=False)
    parser.add_argument('--cfg_scale', type=float, default=3.0)
    parser.add_argument('--embed_upscale', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    
    parser.add_argument('--max_len', type=int, default=4096)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--wandb', action='store_true', default=False)

    return parser

@torch.no_grad()
def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    _, pred = output.topk(max(topk), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res

def update_metrics(out_head, target_head, loss_mask, top_3acc):
    _, predicted = torch.max(out_head, 2)
    _, target = torch.max(target_head, 2)
    
    total = loss_mask.sum().item()
    correct = ((predicted == target) * loss_mask.squeeze()).sum().item()

    out_head_flat = out_head.reshape(-1, target_head.shape[-1])[loss_mask.reshape(-1) == 1]
    target_flat = target.reshape(-1)[loss_mask.reshape(-1) == 1]
    
    topkacc = top_accuracy(out_head_flat, target_flat, (1, 2, 3))

    for top_i in range(len(topkacc)):
        top_3acc[top_i] += topkacc[top_i]

    return correct, total

def log_metrics(optimizer, ploss, vloss, loss, correct, total, top_3acc, phase, wandb):
    
    logdict = {
        f"{phase}/lr": optimizer.param_groups[0]["lr"] if phase == "train" else None,
        f"{phase}/vloss": vloss.item() if vloss is not None else None,
        f"{phase}/ploss": ploss.item() if ploss is not None else None,
        f"{phase}/loss": loss.item(),
        f"{phase}/acc": correct / total
    }

    for id, i in enumerate(top_3acc):
        logdict[f'{phase}/top_{id + 1}_acc'] = i.item() / total
    if wandb:
        wandb.log(logdict)

def run_epoch(args, model, data_loader, optimizer, scheduler, criterion, head, accelerator, is_warmup, train_mode=True):
    model.train() if train_mode else model.eval()
    
    top_3acc = [0 for _ in range(3)]
    correct, total = 0, 0
    epoch_loss = 0
    num_batches = 0

    for data in tqdm(data_loader):
        with torch.set_grad_enabled(train_mode):
            if train_mode:
                optimizer.zero_grad()
            predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            
            with torch.no_grad():
                target_head = head(data["target"])
                if args.cfg_loss:
                    """
                        Note that target_head[::2] is a conditioned logits and target_head[1::2] is an unconditioned logits.
                        Although the original formula for the CFG is cond + scale * (cond - uncond), we found that
                        the official implementation of Lumina-mGPT uses uncond + scale * (cond - uncond) instead and
                        thus we follow the same implementation. (This is equivalent to cond + (scale-1) * (cond - uncond)).
                        
                        Note that here the size of target_head is half of the original target_head.
                    """

                    target_head = target_head[::2] + args.cfg_scale * (target_head[::2] - target_head[1::2])
                    
                target_p = nn.Softmax(dim=2)(target_head).detach()
                
            
            out_head = head(predict)
            if args.cfg_loss:
                out_head = out_head[::2] + args.cfg_scale * (out_head[::2] - out_head[1::2])
            
            out_logp = nn.LogSoftmax(dim=2)(out_head)
            loss_mask = data["loss_mask"][:, :, None]
            if args.cfg_loss:
                p_loss_mask = loss_mask[::2]
            else:
                p_loss_mask = loss_mask

            plogp = target_p * out_logp
            ploss = -torch.sum(torch.sum(p_loss_mask * plogp, 2)) / (p_loss_mask.sum() + 1e-5)
            vloss = torch.sum(torch.mean(loss_mask * criterion(predict, data["target"]), 2)) / (loss_mask.sum() + 1e-5)
            loss = vloss + args.p_w * ploss

            if train_mode:
                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                if is_warmup:
                    scheduler.step()

        with torch.no_grad():
            if not args.cfg_loss and args.coupled:
                """
                    Even when the CFG loss is not used, we need to use CFG for accuracy calculation in the coupled setting.
                    Note that if the dataset is not coupled, CFG for accuracy calculation is not available.
                """
                target_head = target_head[::2] + args.cfg_scale * (target_head[::2] - target_head[1::2])
                out_head = out_head[::2] + args.cfg_scale * (out_head[::2] - out_head[1::2])
                p_loss_mask = loss_mask[::2]

            correct_batch, total_batch = update_metrics(out_head, target_head, p_loss_mask, top_3acc)
            correct += correct_batch
            total += total_batch

        if accelerator.is_main_process and loss_mask.sum().item() != 0 and train_mode:
            log_metrics(optimizer, ploss, vloss, loss, correct, total, top_3acc, "train", args.wandb)

        epoch_loss += loss.item()
        num_batches += 1

    correct = torch.tensor(correct, dtype=torch.float32).to(accelerator.device)
    total = torch.tensor(total, dtype=torch.float32).to(accelerator.device)
    epoch_loss = torch.tensor(epoch_loss, dtype=torch.float32).to(accelerator.device)

    correct, total, epoch_loss = accelerator.gather_for_metrics((correct, total, epoch_loss))
    correct = correct.sum().item()
    total = total.sum().item()
    epoch_loss = epoch_loss.mean()

    top_3acc = [accelerator.gather_for_metrics(torch.tensor(acc, dtype=torch.float32).to(accelerator.device)).sum() for acc in top_3acc]

    return epoch_loss / num_batches, correct, total, top_3acc

def run_train_drafter(args):
    if args.cfg_loss and not args.coupled:
        raise ValueError("--cfg_loss can not be activated without --coupled.")

    set_seed(0)
    accelerator = Accelerator(
                    mixed_precision='bf16',
                    gradient_accumulation_steps=args.gradient_accumulation_steps,)
    
    if accelerator.is_main_process:
        if args.wandb:
            wandb.login(key="726be770e2a351a53a5aab7e7f7772dfc603a233")

        run_name = f"{args.model}_lr{args.lr}_p_w{args.p_w}_bsz{args.bs}_gradacc_{args.gradient_accumulation_steps}"
        run_name += f"_epochs{args.num_epochs}"
        if args.coupled:
            run_name += "_coupled"
        if args.cfg_loss:
            run_name += f"_cfgloss_cfgscale_{args.cfg_scale}"
        if args.embed_upscale > 1.0:
            run_name += f"_embed_upscale_{args.embed_upscale}"
        run_name += "_mscoco2017train30k"
        if args.wandb:
            wandb.init(project="eagle-lumina-mGPT", name=run_name, config=args)
    if args.model == "lumina_mgpt":
        from models.configs.configuration_lumina_mgpt import ChameleonConfig
        from models.drafters.cnets_lumina_mgpt import Model
        base_config = ChameleonConfig.from_pretrained(args.base_path)
    elif args.model == "anole":
        from models.configs.configuration_anole import ChameleonConfig
        from models.drafters.cnets_anole import Model
        base_config = ChameleonConfig.from_pretrained(args.base_path)
    elif "llamagen" in args.model:
        from transformers import AutoConfig
        from models.drafters.cnets_llamagen import Model
        base_config = AutoConfig.from_pretrained(args.base_path)
    else:
        raise ValueError("Invalid model name.")

    ### LOAD `lm_head` ########################################################################
    head = torch.nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)

    try:
        with open(os.path.join(args.base_path, "model.safetensors.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        with safe_open(os.path.join(args.base_path, head_path),
                    framework="pt",
                    device="cpu") as f:
            tensor_slice = f.get_slice("lm_head.weight")
            _, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except:
        try:
            head_path = "model.safetensors"
            with safe_open(os.path.join(args.base_path, head_path),
                        framework="pt",
                        device="cpu") as f:
                tensor_slice = f.get_slice("lm_head.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim].float()
        except:
            head_path = "pytorch_model.bin"
            weights = torch.load(os.path.join(args.base_path, head_path), weights_only=True)
            tensor = weights["lm_head.weight"].float()
    head.weight.data = tensor
    head.eval()

    for param in head.parameters():
        param.requires_grad = False
    ###########################################################################################

    if args.data_noise == "uniform":
        aug = AddUniformNoise(std=args.std)
    elif args.data_noise == "gaussian":
        aug = AddGaussianNoise(mean=args.mean, std=args.std)
    else:
        aug = None

    data_path = list_files(args.data_dir)

    train_data_path = data_path[:int(len(data_path) * args.train_data_ratio)]
    test_data_path = data_path[int(len(data_path) * args.train_data_ratio):]

    if args.coupled:
        if args.model != "lumina_mgpt":
            raise ValueError("--coupled can only be used with lumina_mgpt model.")
        train_dataset = CoupledDataset(train_data_path, max_len=args.max_len, transform=aug)
        test_dataset = CoupledDataset(test_data_path, max_len=args.max_len)

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                collate_fn=DataCollatorWithPaddingForCoupled(), num_workers=0,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                collate_fn=DataCollatorWithPaddingForCoupled(), num_workers=0, pin_memory=True)
        
    else:
        train_dataset = CustomDataset(train_data_path, max_len=args.max_len, transform=aug, model=args.model)
        test_dataset = CustomDataset(test_data_path, max_len=args.max_len, model=args.model)

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                collate_fn=DataCollatorWithPadding(), num_workers=0,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                collate_fn=DataCollatorWithPadding(), num_workers=0, pin_memory=True)
    
    if accelerator.is_main_process:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    config = EConfig.from_pretrained(args.config_path)
    model = Model(config, load_emb=True, path=args.base_path)

    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    if args.is_warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps_ratio * len(train_loader),
                                                    num_training_steps=args.num_epochs * len(train_loader))

        model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
            model, head, optimizer, train_loader, test_loader, scheduler
        )
    else:
        model, head, optimizer, train_loader, test_loader = accelerator.prepare(
            model, head, optimizer, train_loader, test_loader
        )
 
    for epoch in range(args.num_epochs):
        epoch_loss, epoch_correct, epoch_total, epoch_top3\
            = run_epoch(args, model, train_loader, optimizer, scheduler, criterion, head, accelerator, args.is_warmup, train_mode=True)
        
        if accelerator.is_main_process:
            log_metrics(optimizer, None, None, epoch_loss, epoch_correct, epoch_total, epoch_top3, "epoch", args.wandb)
        
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.num_epochs:
            test_loss, test_correct, test_total, test_top3\
            = run_epoch(args, model, test_loader, optimizer, scheduler, criterion, head, accelerator, args.is_warmup, train_mode=False)
            
            if accelerator.is_main_process:
                log_metrics(optimizer, None, None, test_loss, test_correct, test_total, test_top3, "test", args.wandb)

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.num_epochs:
            if accelerator.is_main_process:
                accelerator.save_state(output_dir=f"{args.save_dir}/{run_name}/state_{epoch + 1}")

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    
    run_train_drafter(args)

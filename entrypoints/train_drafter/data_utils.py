import os
import random
import torch

from typing import Any, Dict, List
from torch.utils.data import Dataset

def list_files(path):
    datapath = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def inject_noise(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        return noisy_tensor

    def __call__(self, data):
        data["hidden_states"] = self.inject_noise(data["hidden_states"])
        return data

class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def inject_noise(self, tensor):
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        return noisy_tensor

    def __call__(self, data):
        data["hidden_states"] = self.inject_noise(data["hidden_states"])
        return data

class CustomDataset(Dataset):
    def __init__(self, datapath, max_len, transform=None, model=None):
        self.model = model
        self.data = datapath
        self.max_len = max_len
        self.transform = transform
        if model == "lumina_mgpt":
            self.num_image_tokens = 2357
        elif model == "anole":
            self.num_image_tokens = 1024
        elif model == "llamagen":
            self.num_image_tokens = 256
        elif model == "llamagen2":
            self.num_image_tokens = 1024
        else:
            raise ValueError("Invalid model name")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index], weights_only=True)
        
        if self.model == "lumina_mgpt" or self.model == "anole":

            if random.random() < 0.9:
                # 90% probability to use cond_input_ids and cond_hidden_states
                input_ids = data['cond_input_ids'][:self.max_len][None, :]
                hidden_states = data['cond_hidden_states'][:self.max_len][None, :]
            else:
                # 10% probability to use uncond_input_ids and uncond_hidden_states
                input_ids = data['uncond_input_ids'][:self.max_len][None, :]
                hidden_states = data['uncond_hidden_states'][:self.max_len][None, :]
            
            loss_mask = torch.ones_like(input_ids)
            if input_ids.shape[1] > self.num_image_tokens:
                # cond_input_ids
                loss_mask[:, :-self.num_image_tokens] = 0
            else:
                # uncond_input_ids
                pass

            attention_mask = [1] * input_ids.shape[1]

            # input_ids_targets
            zeropadding = torch.tensor([[0]])
            
            input_ids_target = input_ids[:, :-1]
            input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        elif "llamagen" in self.model:
            hidden_states = data['hidden_state'][:self.max_len][None, :]
            input_ids = data['input_ids'][:self.max_len][None, :]
            input_ids_padding = torch.zeros_like(input_ids)[:, :119]
            input_ids = torch.cat((input_ids_padding, input_ids), dim=1)
            loss_mask = data["loss_mask"][:self.max_len][None, :]

            length = hidden_states.shape[1]
            attention_mask = [1] * length
            
            zeropadding = torch.tensor([[0]])
            
            input_ids_target = input_ids
            input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0
        
        target = hidden_states[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        item = {
            "input_ids": input_ids_target,
            "hidden_states": hidden_states,
            "target": target,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

        if self.transform:
            item = self.transform(item)

        return item

class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_states'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_states'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch

class CoupledDataset(Dataset):
    def __init__(self, datapath, max_len, transform=None):
        self.data = datapath
        self.max_len = max_len
        self.transform = transform
        self.num_image_tokens = 2356 # same for all the 768 x 768 resolution images

    def __len__(self):
        return len(self.data)

    def prepare_data(self, data, conditioned=True):
        if conditioned:
            prefix = "cond"
        else:
            prefix = "uncond"
            cond_len = len(data['cond_input_ids'])
        
        if not conditioned:
            ids_zeropadding = torch.zeros(cond_len - self.num_image_tokens, dtype=torch.int)
            data['uncond_input_ids'] = torch.cat([ids_zeropadding, data['uncond_input_ids']])

            hidden_states_zeropadding = torch.zeros(cond_len - self.num_image_tokens, data['uncond_hidden_states'].shape[1], dtype=torch.float)
            data['uncond_hidden_states'] = torch.cat([hidden_states_zeropadding, data['uncond_hidden_states']], dim=0)

        input_ids = data[f"{prefix}_input_ids"][:self.max_len][None, :]
        hidden_states = data[f"{prefix}_hidden_states"][:self.max_len][None, :]

        loss_mask = torch.ones_like(input_ids)
        loss_mask[:, :-self.num_image_tokens] = 0
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0 # The last token has no next token to predict

        attention_mask = [1] * input_ids.shape[1]

        # input_ids_targets
        zeropadding = torch.tensor([[0]])
        input_ids_target = input_ids[:, :-1]
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        # hidden_state targets
        target = hidden_states[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)

        item = {
            "input_ids": input_ids_target,
            "hidden_states": hidden_states,
            "target": target,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

        return item


    def __getitem__(self, index):
        data = torch.load(self.data[index])
        cond_item = self.prepare_data(data, conditioned=True)
        uncond_item = self.prepare_data(data, conditioned=False)

        if self.transform:
            cond_item = self.transform(cond_item)
            uncond_item = self.transform(uncond_item)

        item = {
            "cond": cond_item,
            "uncond": uncond_item,
        }

        return item

class DataCollatorWithPaddingForCoupled:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(
            max(item['cond']['hidden_states'].shape[1] for item in features),
            max(item['uncond']['hidden_states'].shape[1] for item in features)
        )

        all_items = []
        for item in features:
            all_items.append(item['cond'])
            all_items.append(item['uncond'])

        batch = {
            "input_ids": torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in all_items]),
            "hidden_states": torch.cat([self.paddingtensor(item['hidden_states'], max_length) for item in all_items]),
            "target": torch.cat([self.paddingtensor(item['target'], max_length) for item in all_items]),
            "loss_mask": torch.tensor(
                [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in all_items]
            ),
            "attention_mask": torch.tensor(
                [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in all_items]
            ),
        }

        return batch
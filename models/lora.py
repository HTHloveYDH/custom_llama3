import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


'''reference url: https://github.com/hkproj/pytorch-lora'''
class LoRAParametrization(nn.Module):
    def __init__(self, layer:nn.Module, rank=1, alpha=1, device='cpu'):
        super().__init__()
        features_in, features_out = layer.weight.shape
        # Section 4.1 of the paper: 
        # We use a random Gaussian initialization for A and zero for B, so ∆W = BA is zero at the beginning of training
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        # Section 4.1 of the paper: 
        #   We then scale ∆Wx by α/r , where α is a constant in r. 
        #   When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately. 
        #   As a result, we simply set α to the first r we try and do not tune it. 
        #   This scaling helps to reduce the need to retune hyperparameters when we vary r.
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B*A)*scale
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights

    @staticmethod
    def get_original_weights(net:nn.Module):
        original_weights = {}
        for name, param in net.named_parameters():
            original_weights[name] = param.clone().detach()  
        return original_weights 
    
    @staticmethod
    def count_original_non_lora_weights(net:nn.Module):
        # Print the size of the weights matrices of the network
        # Save the count of the total number of weights
        non_lora_weights = 0
        for index, module in enumerate(net.modules()):
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    non_lora_weights += module.weight.nelement() + module.bias.nelement()
                    print(f'module {index+1}: W: {module.weight.shape} + B: {module.bias.shape}')
                else:
                    non_lora_weights += module.weight.nelement()
                    print(f'module {index+1}: W: {module.weight.shape} + B: None')
        print(f'Total number of weights: {non_lora_weights:,}')
        return non_lora_weights
    
    @classmethod
    def inject_lora_weights(cls, layer:nn.Module, rank=1, alpha=1):
        parametrize.register_parametrization(layer, "weight", cls(layer, rank, alpha))

    @staticmethod
    def count_lora_weights(net:nn.Module, original_non_lora_weights:int):
        lora_weights = 0
        non_lora_weights = 0
        for index, module in enumerate(net.modules()):
            if isinstance(module, nn.Linear):
                lora_weights += module.parametrizations["weight"][0].lora_A.nelement() + module.parametrizations["weight"][0].lora_B.nelement()
                if module.bias is not None:
                    non_lora_weights += module.weight.nelement() + module.bias.nelement()
                    print(
                        f'module {index+1}: W: {module.weight.shape} + B: {module.bias.shape} + Lora_A: {module.parametrizations["weight"][0].lora_A.shape} + Lora_B: {module.parametrizations["weight"][0].lora_B.shape}'
                    )
                else:
                    non_lora_weights += module.weight.nelement()
                    print(
                        f'module {index+1}: W: {module.weight.shape} + Lora_A: {module.parametrizations["weight"][0].lora_A.shape} + Lora_B: {module.parametrizations["weight"][0].lora_B.shape}'
                    )
        # The non-LoRA weights count must match the original network
        assert non_lora_weights == original_non_lora_weights
        print(f'Total number of weights (original): {non_lora_weights:,}')
        print(f'Total number of weights (original + LoRA): {lora_weights + non_lora_weights:,}')
        print(f'Weights introduced by LoRA: {lora_weights:,}')
        weights_incremment = (lora_weights / non_lora_weights) * 100
        print(f'Weights incremment: {weights_incremment:.3f}%')
        return lora_weights, non_lora_weights
    
    @staticmethod        
    def enable_disable_lora(net:nn.Module, enabled=True):
        for module in net.modules():
            if isinstance(module, nn.Linear):
                module.parametrizations["weight"][0].enabled = enabled

    @staticmethod
    def freeze_non_lora_weights(net:nn.Module):
        # Freeze the non-Lora weights
        for name, param in net.named_parameters():
            if 'lora' not in name:
                print(f'Freezing non-LoRA parameter {name}')
                param.requires_grad = False
    
    @staticmethod        
    def confirm_original_weights(net:nn.Module, original_weights:dict):
        for name, module in net.named_modules():
            if isinstance(module, nn.Linear):
                # Check that the frozen weights are still unchanged by the finetuning
                assert torch.all(module.parametrizations.weight.original == original_weights[f'{name}.weight'])
        LoRAParametrization.enable_disable_lora(enabled=True)
        # The new module is obtained by the "forward" function of our LoRA parametrization
        # The original weights have been moved to module.parametrizations.weight.original
        # More info here: https://pytorch.org/tutorials/intermediate/parametrizations.html#inspecting-a-parametrized-module
        for name, module in net.named_modules():
            if isinstance(module, nn.Linear):
                assert torch.equal(
                    module.weight, 
                    module.parametrizations.weight.original + (module.parametrizations.weight[0].lora_B @ module.parametrizations.weight[0].lora_A) * module.parametrizations.weight[0].scale
                )
        LoRAParametrization.enable_disable_lora(enabled=False)
        # If we disable LoRA, the module is the original one
        for name, module in net.named_modules():
            if isinstance(module, nn.Linear):
                assert torch.equal(module.weight, original_weights[f'{name}.weight'])

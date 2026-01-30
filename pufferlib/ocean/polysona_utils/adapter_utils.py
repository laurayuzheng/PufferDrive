import torch
from torch import nn
import loralib as lora

ATTENTION_LINEARS = [
    'ca_',
    'sa_',
    'out_proj',
    'linear',
] + [
    'to_q', 
    'to_k', 
    'to_v', 
    'to_s', 
    'to_g', 
    'to_out'
]

BLACKLIST = [
    'to_k_r', 
    'to_v_r'
]


def _name_is_attention(name):
    return any(
        [att in name for att in ATTENTION_LINEARS]
    )  # and ("out_proj" not in name)


def _name_is_not_blacklisted(name):
    return all([substr not in name for substr in BLACKLIST])


def replace_layers_simple_lora(model, r=8, only_attention=True):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers_simple_lora(module, only_attention=only_attention)

        name_is_attention = _name_is_attention(name)
        name_is_not_blacklisted = _name_is_not_blacklisted(name)
        valid = (
            name_is_attention if only_attention else True
        ) and name_is_not_blacklisted

        if isinstance(module, nn.Linear) and valid:
            new_linear = lora.Linear(module.in_features, module.out_features, r=r)

            if module.weight is not None:
                with torch.no_grad():
                    new_linear.weight.copy_(module.weight)
            if module.bias is not None:
                with torch.no_grad():
                    new_linear.bias.copy_(module.bias)
            setattr(model, name, new_linear)


def replace_layers(
    model, adapter_class, n_tasks, n_skills, skills, rank=16, only_attention=True
):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(
                module,
                adapter_class,
                n_tasks,
                n_skills,
                skills,
                rank=rank,
                only_attention=only_attention,
            )

        name_is_attention = _name_is_attention(name)
        name_is_not_blacklisted = _name_is_not_blacklisted(name)
        valid = (
            name_is_attention if only_attention else True
        ) and name_is_not_blacklisted

        if (
            isinstance(module, nn.Linear) and valid
        ):  #  and "ca_" not in name and "sa_" not in name
            new_linear = adapter_class(
                n_tasks, n_skills, skills, module.weight, module.bias, r=rank
            )
            setattr(model, name, new_linear)


class PersonaReshapeTracker:
    """Tracks persona logits and applies the same reshaping operations as input tensors."""
    
    def __init__(self, initial_personas, batch_size, num_steps=None):
        self.current_personas = initial_personas
        self.batch_size = batch_size
        self.num_steps = num_steps
        
    def reshape(self, *shape):
        """Apply reshape to persona logits matching the input tensor reshape."""
        if self.current_personas is None:
            return self.current_personas
            
        # Get current persona shape
        current_shape = self.current_personas.shape
        num_personas = current_shape[-1]  # Last dimension is always persona count
        
        # Handle -1 in reshape (infer dimension)  
        new_shape = list(shape)
        if -1 in new_shape:
            # For persona logits, we only care about the leading dimension
            # The second dimension should always be num_personas
            if len(new_shape) == 2 and new_shape[1] == -1:
                new_shape[1] = num_personas
            elif len(new_shape) == 2 and new_shape[0] == -1:
                # Calculate leading dimension: total_elements / num_personas  
                total_elements = current_shape[0] * current_shape[1]
                new_shape[0] = total_elements // new_shape[1]
        
        # Only reshape if dimensions actually changed
        target_leading_dim = new_shape[0] if len(new_shape) > 0 else current_shape[0]
        if target_leading_dim != current_shape[0]:
            # Reshape preserving the persona dimension
            self.current_personas = self.current_personas.view(target_leading_dim, num_personas)
        
        return self.current_personas
    
    def transpose(self, dim0, dim1):
        """Apply transpose to persona logits if it affects the leading dimension."""
        if self.current_personas is None or dim0 != 0 and dim1 != 0:
            return self.current_personas
            
        # If transpose involves the leading dimension, apply it
        if dim0 == 0 or dim1 == 0:
            # For persona logits, we only care about reordering the leading dimension
            # Need to reshape to match the transposed structure
            current_shape = self.current_personas.shape
            if len(current_shape) == 2:  # [leading_dim, num_personas]
                # Infer the 2D structure to transpose
                if self.num_steps is not None and current_shape[0] == self.batch_size * self.num_steps:
                    # Currently [batch*time, personas] -> reshape to [batch, time, personas] -> transpose -> reshape back
                    temp = self.current_personas.view(self.batch_size, self.num_steps, -1)
                    temp = temp.transpose(dim0, dim1)
                    self.current_personas = temp.reshape(-1, temp.shape[-1])
                elif self.num_steps is not None and current_shape[0] == self.num_steps * self.batch_size:
                    # Currently [time*batch, personas] -> reshape to [time, batch, personas] -> transpose -> reshape back  
                    temp = self.current_personas.view(self.num_steps, self.batch_size, -1)
                    temp = temp.transpose(dim0, dim1)
                    self.current_personas = temp.reshape(-1, temp.shape[-1])
        
        return self.current_personas
    
    def repeat_interleave(self, repeats, dim=0):
        """Apply repeat_interleave to persona logits."""
        if self.current_personas is None:
            return self.current_personas
        self.current_personas = self.current_personas.repeat_interleave(repeats, dim=dim)
        return self.current_personas
    
    @staticmethod
    def set_layer_personas(layer_or_module_list, personas):
        """Set personas for all SkilledModule layers within a module or module list."""
        def _recursive_set(module):
            for name, child in module.named_children():
                if len(list(child.children())) > 0:
                    _recursive_set(child)
                if hasattr(child, 'task_ids'):  # SkilledModule or similar
                    child.task_ids = personas
        
        if hasattr(layer_or_module_list, '__iter__'):
            # Handle ModuleList 
            _recursive_set(layer_or_module_list)
        else:
            _recursive_set(layer_or_module_list)


def inform_layers_with_tracker(model, adapter_class, persona_tracker):
    """Inform layers using the current state of the persona tracker."""
    def _recursive_inform(module):
        for name, child in module.named_children():
            if len(list(child.children())) > 0:
                _recursive_inform(child)
            if isinstance(child, adapter_class):
                child.task_ids = persona_tracker.current_personas
    
    if hasattr(model, '__iter__'):
        # Handle ModuleList or similar
        for module in model:
            _recursive_inform(module)
    else:
        _recursive_inform(model)


def inform_layers(model, adapter_class, value):
    """Legacy function - kept for backward compatibility"""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            inform_layers(module, adapter_class, value)

        if isinstance(module, adapter_class):
            module.task_ids = value

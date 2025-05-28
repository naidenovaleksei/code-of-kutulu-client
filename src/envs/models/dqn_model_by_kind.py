import torch
import torch.nn as nn

class DQNExtByKind(nn.Module):
    def __init__(self, num_dirs, features_dim, hidden_dim, inner_dim, num_classes, entity_kinds=None):
        super(DQNExtByKind, self).__init__()
        
        # Default entity kinds if not provided
        if entity_kinds is None:
            entity_kinds = ["EXPLORER", "WANDERER", "SLASHER", "EFFECT_PLAN", "EFFECT_LIGHT", "EFFECT_SHELTER", "EFFECT_YELL"]
        
        self.entity_kinds = entity_kinds
        
        # Create separate towers for each entity kind
        self.features_towers = nn.ModuleDict({
            kind: nn.Sequential(
                nn.Linear(features_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, inner_dim)
            ) for kind in entity_kinds
        })
        
        self.dir_towers = nn.ModuleDict({
            kind: nn.Sequential(
                nn.Linear(num_dirs, inner_dim),
                nn.ReLU()
            ) for kind in entity_kinds
        })
        
        # Combined processing for each kind
        self.entity_towers = nn.ModuleDict({
            kind: nn.Sequential(
                nn.Linear(inner_dim * 2, inner_dim),
                nn.ReLU()
            ) for kind in entity_kinds
        })
        
        # Impact layers for each kind
        self.entity_impact_towers = nn.ModuleDict({
            kind: nn.Linear(inner_dim, num_classes) for kind in entity_kinds
        })
        
        # Final output layer
        self.out_linear = nn.Linear(num_classes * len(entity_kinds), num_classes)
        
        self.num_classes = num_classes
        self.inner_dim = inner_dim
        self.num_dirs = num_dirs
        
    def forward(self, data_by_kind):
        """
        Process each entity kind through its own tower and combine the results.
        
        Args:
            data_by_kind: Dictionary where keys are entity kinds and values are dictionaries with
                         'entity_features' and 'entity_dir' keys
        
        Returns:
            Tensor of shape [batch_size, num_classes] containing the output logits
        """
        batch_size = None
        kind_outputs = []
        
        # Process each kind through its own tower
        for kind in self.entity_kinds:
            if kind in data_by_kind:
                data = data_by_kind[kind]
                entity_features = data['entity_features']
                entity_dir = data['entity_dir']
                
                # Set batch_size if not already set
                if batch_size is None:
                    if len(entity_features.shape) > 2:
                        batch_size = entity_features.shape[0]
                    else:
                        batch_size = 1
                        entity_features = entity_features.unsqueeze(0)
                        entity_dir = entity_dir.unsqueeze(0)
                
                # Create mask for valid entities
                entities_mask = (entity_dir > 0).max(dim=-1, keepdim=True)[0]
                
                # Process features through the kind's tower
                x_features = self.features_towers[kind](entity_features)
                x_dir = self.dir_towers[kind](entity_dir)
                
                # Combine features and directions
                x_combined = torch.cat((x_features, x_dir), dim=-1)
                
                # Process through entity tower
                x = self.entity_towers[kind](x_combined)
                
                # Apply mask and aggregate
                x_masked = x * entities_mask
                
                # If there are no valid entities of this kind, use zeros
                if entities_mask.sum() > 0:
                    # Average over entities of this kind
                    x_agg = x_masked.sum(dim=1) / entities_mask.sum(dim=1)
                else:
                    x_agg = torch.zeros(batch_size, self.inner_dim, device=x.device)
                
                # Get impact of this kind on the output
                kind_output = self.entity_impact_towers[kind](x_agg)
                kind_outputs.append(kind_output)
            else:
                # If this kind is not present, use zeros
                if batch_size is None:
                    batch_size = 1
                
                kind_output = torch.zeros(batch_size, self.num_classes, device=next(self.parameters()).device)
                kind_outputs.append(kind_output)
        
        # Combine outputs from all kinds
        if len(kind_outputs) == 0:
            # Handle the case where there are no entities
            return torch.zeros(batch_size or 1, self.num_classes, device=next(self.parameters()).device)
        
        # Concatenate all kind outputs
        combined_output = torch.cat(kind_outputs, dim=1)
        
        # Final output layer
        output = self.out_linear(combined_output)
        
        return output

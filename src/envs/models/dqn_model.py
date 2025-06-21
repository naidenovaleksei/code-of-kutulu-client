import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, num_classes, vocab_size=7, embed_dim=32, hidden_dim=32):
        super(DQN, self).__init__()
        self.edist_embs = nn.Embedding(vocab_size, embed_dim)
        self.wdist_embs = nn.Embedding(vocab_size, embed_dim)
        
        self.num_classes = num_classes
        self.edir_linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
            # nn.ReLU()
        )
        self.wdir_linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
            # nn.ReLU()
        )
        # self.output = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, num_classes),
        # )
        # self.softmax = nn.Softmax(dim=-1)
    
    def get_policy(self, x):
        return self.forward(x)

    def forward(self, data):
        assert data['closest_explorer_dir'].shape[-1] == self.num_classes
        assert data['closest_wanderer_dir'].shape[-1] == self.num_classes

        edist_dir = (data['closest_explorer_dir'] * (data['closest_explorer_dist'] + 1))
        wdist_dir = (data['closest_wanderer_dir'] * (data['closest_wanderer_dist'] + 1))
        
        edist_dir_masked = (self.edist_embs(edist_dir) * (edist_dir.unsqueeze(-1) > 0))
        wdist_dir_masked = (self.wdist_embs(wdist_dir) * (wdist_dir.unsqueeze(-1) > 0))
        
        ex = self.edir_linear(edist_dir_masked).squeeze(-1)
        wx = self.wdir_linear(wdist_dir_masked).squeeze(-1)

        return ex + wx


class DuelingDQN(nn.Module):
    def __init__(self, num_classes, vocab_size=7, embed_dim=32, hidden_dim=32):
        super(DuelingDQN, self).__init__()
        # Shared embedding layers (same as DQN)
        self.edist_embs = nn.Embedding(vocab_size, embed_dim)
        self.wdist_embs = nn.Embedding(vocab_size, embed_dim)
        
        self.num_classes = num_classes
        
        # Shared feature extraction layers
        self.edir_shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU()
        )
        self.wdir_shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream - outputs single scalar V(s)
        self.edir_value = nn.Linear(hidden_dim, 1)
        self.wdir_value = nn.Linear(hidden_dim, 1)
        
        # Advantage stream - outputs advantage for each action A(s,a)
        self.edir_advantage = nn.Linear(hidden_dim, 1)
        self.wdir_advantage = nn.Linear(hidden_dim, 1)
    
    def get_policy(self, x):
        return self.forward(x)

    def forward(self, data):
        assert data['closest_explorer_dir'].shape[-1] == self.num_classes
        assert data['closest_wanderer_dir'].shape[-1] == self.num_classes

        edist_dir = (data['closest_explorer_dir'] * (data['closest_explorer_dist'] + 1))
        wdist_dir = (data['closest_wanderer_dir'] * (data['closest_wanderer_dist'] + 1))
        
        edist_dir_masked = (self.edist_embs(edist_dir) * (edist_dir.unsqueeze(-1) > 0))
        wdist_dir_masked = (self.wdist_embs(wdist_dir) * (wdist_dir.unsqueeze(-1) > 0))
        
        # Shared feature extraction
        e_shared = self.edir_shared(edist_dir_masked)
        w_shared = self.wdir_shared(wdist_dir_masked)
        
        # Value stream - single value per state
        e_value = self.edir_value(e_shared).squeeze(-1)  # [batch_size, num_classes]
        w_value = self.wdir_value(w_shared).squeeze(-1)  # [batch_size, num_classes]
        
        # Advantage stream - advantage per action
        e_advantage = self.edir_advantage(e_shared).squeeze(-1)  # [batch_size, num_classes]
        w_advantage = self.wdir_advantage(w_shared).squeeze(-1)  # [batch_size, num_classes]
        
        # Combine explorer and wanderer streams
        value = e_value + w_value  # [batch_size, num_classes]
        advantage = e_advantage + w_advantage  # [batch_size, num_classes]
        
        # Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        # Subtract mean advantage to ensure identifiability
        advantage_mean = advantage.mean(dim=-1, keepdim=True)  # [batch_size, 1]
        q_values = value + advantage - advantage_mean  # [batch_size, num_classes]
        
        return q_values

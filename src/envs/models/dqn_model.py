import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(DQN, self).__init__()
        self.edir_embs = nn.Embedding(vocab_size, embed_dim)
        self.wdir_embs = nn.Embedding(vocab_size, embed_dim)
        self.edir_linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            # nn.ReLU()
        )
        self.wdir_linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            # nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, num_classes),
        )
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, data):
        edir_embs = self.edir_embs(data['closest_explorer_dir'])
        mask = data['closest_explorer_dir'] > 0
        mask_float = mask.float().unsqueeze(-1)
        masked_emb = edir_embs * mask_float
        lengths = mask.sum(dim=1, keepdim=True)
        lengths[lengths == 0] = 1
        edir_embs = masked_emb.sum(dim=1) / lengths
        
        # edir_embs = self.edir_embs(data['closest_explorer_dir'])
        edir_embs = edir_embs / (data['closest_explorer_dist'] + 1)
        edir_x = self.edir_linear(edir_embs)
        # edir_x = edir_x / (data['closest_explorer_dist'])
        
        wdir_embs = self.edir_embs(data['closest_wanderer_dir'])
        mask = data['closest_wanderer_dir'] > 0
        mask_float = mask.float().unsqueeze(-1)
        masked_emb = wdir_embs * mask_float
        lengths = mask.sum(dim=1, keepdim=True)
        lengths[lengths == 0] = 1
        wdir_embs = masked_emb.sum(dim=1) / lengths
        
        # wdir_embs = self.wdir_embs(data['closest_wanderer_dir'])
        wdir_embs = wdir_embs / (data['closest_wanderer_dist'] + 1)
        wdir_x = self.wdir_linear(wdir_embs)
        # wdir_x = wdir_x / (data['closest_wanderer_dist'])

        x = torch.cat((edir_x, wdir_x), dim=-1)
        x = self.output(x)
        x = torch.clamp(x, -100., 100.)

        return x

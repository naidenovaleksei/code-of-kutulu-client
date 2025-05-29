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

import torch
import torch.nn as nn

class CrossEntropyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(CrossEntropyModel, self).__init__()
        self.edir_embs = nn.Embedding(vocab_size, embed_dim)
        self.wdir_embs = nn.Embedding(vocab_size, embed_dim)
        self.edir_linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU()
        )
        self.wdir_linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU()
        )
        self.output = nn.Linear(hidden_dim * 2, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, data):
        edir_embs = self.edir_embs(data['closest_explorer_dir'] + 1).sum(1)
        edir_x = self.edir_linear(edir_embs)
        edir_x = edir_x / (data['closest_explorer_dist'] + 1)
        
        wdir_embs = self.wdir_embs(data['closest_wanderer_dir'] + 1).sum(1)
        wdir_x = self.wdir_linear(wdir_embs)
        wdir_x = wdir_x / (data['closest_wanderer_dist'] + 1)

        x = torch.cat((edir_x, wdir_x), dim=-1)
        x = self.output(x)

        return x

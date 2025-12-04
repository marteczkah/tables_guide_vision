import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
    
class MultiplePairsContrastiveLoss(nn.Module):
    def __init__(self,  device, similarity_thr = 0.05, temperature=0.1, morphological=False):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.ind_continuous = [i for i in range(13)]
        self.ind_continuous.append(23)
        self.ind_binary = [i for i in range(13, 23)]
        self.similarity_thr = similarity_thr
        self.morphological = morphological            
    
    def forward(self, features, continuous, categorical, len_categorical=336, lam=0.5):
        if self.morphological:
            distance_matrix = torch.cdist(continuous, continuous, p=2)
            similarity =  1 / (1 + distance_matrix)
        else:
            distance_matrix = torch.cdist(continuous, continuous, p=2)
            continuous_similarity =  1 / (1 + distance_matrix)
            continuous_similarity = continuous_similarity*2-1 
            categorical_similarity = torch.matmul(categorical, categorical.T) / len_categorical
            similarity = lam * continuous_similarity + (1 - lam) * categorical_similarity
        similarity.fill_diagonal_(0)
        mask = torch.zeros_like(similarity).to(self.device)
        for i, sim in enumerate(similarity):
            max_sim = sim.max()  
            positive_indices = (sim >= max_sim - self.similarity_thr).nonzero(as_tuple=True)[0]
            mask[i, positive_indices] = 1
        # row_sums = mask.sum(dim=1)  # if you want to know the num of pairs 

        features = F.normalize(features, dim=1)
        logits = torch.einsum("nc,mc->nm", features, features) / self.temperature

        exp_sim = torch.exp(logits).to(self.device) 
        positives = exp_sim * mask  
        loss = -torch.log(positives.sum(dim=1) / exp_sim.sum(dim=1))

        return loss.mean()
 
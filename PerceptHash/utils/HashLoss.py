import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class TripletLossForHashing(torch.nn.Module):
    """
    A compact and flexible triplet hashing loss function.
    """
    def __init__(self, alpha, beta, margin=1.0):
        """
        Args:
            alpha (float): Weight for quantization loss.
            beta (float): Weight for balance loss.
            margin (float): Triplet loss margin.
        """
        super(TripletLossForHashing, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        logger.info(f"Initialized TripletLossForHashing with alpha(quant)={self.alpha}, beta(balance)={self.beta}, margin={self.margin}")

    def forward(self, h_anchor, h_positive, h_negative):
        """
        Computes total loss and returns individual components.
        """
        # 1. Contrastive (Triplet) Loss
        dist_pos = torch.sum(torch.pow(h_anchor - h_positive, 2), dim=1)
        dist_neg = torch.sum(torch.pow(h_anchor - h_negative, 2), dim=1)
        contrastive_loss = F.relu(self.margin + dist_pos - dist_neg).mean()

        # 2. Quantization Loss
        all_hashes_for_q = torch.cat([h_anchor, h_positive, h_negative], dim=0)
        quantization_loss = (all_hashes_for_q - torch.tanh(all_hashes_for_q)).pow(2).mean()

        # 3. Balance Loss
        all_hashes_for_b = torch.cat([h_anchor, h_positive, h_negative], dim=0)
        balance_loss = (all_hashes_for_b.mean(dim=0)).pow(2).sum()
        
        total_loss = contrastive_loss + self.alpha * quantization_loss + self.beta * balance_loss
        
        return total_loss, contrastive_loss, quantization_loss, balance_loss
    
class AngularTripletLoss(nn.Module):
    """
    Angular/cosine similarity-based triplet hashing loss.
    Designed for L2-normalized model outputs.
    """
    def __init__(self, alpha, beta, hash_bit, margin=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.hash_bit = hash_bit
        self.margin = margin
        logger.info(f"Initialized AngularTripletLoss with alpha={alpha}, beta={beta}, margin={margin}")

    def forward(self, h_anchor, h_positive, h_negative):
        # Inputs are expected to be unit vectors
        
        # 1. Angular Contrastive Loss
        sim_pos = F.cosine_similarity(h_anchor, h_positive, dim=1)
        sim_neg = F.cosine_similarity(h_anchor, h_negative, dim=1)
        
        contrastive_loss = F.relu(self.margin - sim_pos + sim_neg).mean()
        
        # 2. Quantization Loss
        all_hashes = torch.cat([h_anchor, h_positive, h_negative], dim=0)
        scaled_hashes = all_hashes * (self.hash_bit ** 0.5)
        quantization_loss = (scaled_hashes - torch.tanh(scaled_hashes)).pow(2).mean()
        
        # 3. Balance Loss
        balance_loss = (all_hashes.mean(dim=0)).pow(2).sum()
        
        total_loss = contrastive_loss + self.alpha * quantization_loss + self.beta * balance_loss
        
        return total_loss, contrastive_loss, quantization_loss, balance_loss

class AngularTripletLossNoABL(nn.Module):
    """
    Angular triplet hashing loss without Auxiliary Balance Loss (ABL).
    """
    def __init__(self, alpha, beta, hash_bit, margin=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.hash_bit = hash_bit
        self.margin = margin
        logger.info(f"Initialized AngularTripletLossNoABL with alpha={alpha}, beta={beta}(unused), margin={margin}")

    def forward(self, h_anchor, h_positive, h_negative):
        # Inputs are expected to be unit vectors
        
        # 1. Angular Contrastive Loss
        sim_pos = F.cosine_similarity(h_anchor, h_positive, dim=1)
        sim_neg = F.cosine_similarity(h_anchor, h_negative, dim=1)
        contrastive_loss = F.relu(self.margin - sim_pos + sim_neg).mean()
        
        # 2. Quantization Loss
        all_hashes = torch.cat([h_anchor, h_positive, h_negative], dim=0)
        scaled_hashes = all_hashes * (self.hash_bit ** 0.5)
        quantization_loss = (scaled_hashes - torch.tanh(scaled_hashes)).pow(2).mean()
        
        # 3. No Balance Loss (for ablation study)
        balance_loss = torch.tensor(0.0, device=h_anchor.device)
        
        total_loss = contrastive_loss + self.alpha * quantization_loss
        
        return total_loss, contrastive_loss, quantization_loss, balance_loss
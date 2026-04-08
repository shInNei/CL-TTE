import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.PositionalEncoding import PositionalEncoding1D, PositionalEncodingIndex
from models.loss.contrastive_loss import HardContrastiveLoss
from models.blocks.cl import MSM
from models.blocks.poi import PoiResGatedFilMEncoder

class ContrastiveEncoder(nn.Module):
    def __init__(self, d_model, nhead, r_percentile, dropout=0.1, nlayer=4):
        super().__init__()
        self.r_percentile = r_percentile
        
        self.pos_enc = PositionalEncodingIndex(d_model)
        self.msm = MSM(d_model,nhead,dropout,nlayer)
        
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model)
        )
        self.loss = HardContrastiveLoss()
        
    def create_pos_mask(self, y_true):
        # y_true: (B, T)
        if y_true.dim() == 2:
            y_true = y_true.squeeze(-1)
        y_true = y_true.detach()
        B = y_true.size(0)

        y_true = (y_true - y_true.mean()) / (y_true.std() + 1e-6)
        
        dist = torch.abs(y_true.unsqueeze(0) - y_true.unsqueeze(1))
        mask = ~torch.eye(B, dtype=torch.bool, device=y_true.device)
        r = torch.quantile(dist[mask], self.r_percentile).clamp(min=1e-3)

        # expand to 2B
        # y_all = torch.cat([y_true, y_true], dim=0)
        # dist_all = torch.abs(y_all.unsqueeze(0) - y_all.unsqueeze(1))
        # pos_base = (dist_all <= r).float()
        pos_base = (dist <= r).float()
        pos_base.fill_diagonal_(0)
        

        pos_mask = torch.zeros(2 * B, 2 * B, device=y_true.device)
        # ori to ori
        pos_mask[:B, :B] = pos_base
        # aug inherit ori
        pos_mask[B:, :B] = pos_base
        # identity pair
        idx = torch.arange(B, device=y_true.device)
        pos_mask[idx, idx + B] = 1.0
        pos_mask[idx + B, idx] = 1.0

        return pos_mask
    
    def masked_mean_pooling(self, x, mask):
        # x: (B, T, D)
        # mask: (B, T) with 1 = valid, 0 = padding
        mask = mask.float()
        # sum
        x_sum = torch.einsum('btd,bt->bd', x, mask)
        # count valid tokens
        denom = mask.sum(dim=1, keepdim=True)  # (B, 1)
        # avoid division by zero
        denom = denom.clamp(min=1.0)
        return x_sum / denom
    
    def forward(self,x,x_aug,src_key_padding_mask=None,y_true=None):
        # x: (B, T, D)
        # x_aug: (B, T, D)
        if src_key_padding_mask is None:
            valid_mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        else:
            valid_mask = ~src_key_padding_mask
        # encode
        x_pos = self.pos_enc(x, src_key_padding_mask)
        h = self.msm(x_pos,src_key_padding_mask)
        
        if self.training:
            x_aug_pos = self.pos_enc(x_aug, src_key_padding_mask)
            h_aug = self.msm(x_aug_pos,src_key_padding_mask)
            
            # pooling
            z = self.masked_mean_pooling(h,valid_mask) # (B, D)
            z_aug = self.masked_mean_pooling(h_aug,valid_mask) # (B, D)
            
            # projection
            z_proj = self.proj(z)
            z_aug_proj = self.proj(z_aug)
            
            # concat and normalize        
            z_all = torch.cat([z_proj,z_aug_proj],dim=0) # (2B, D // 2)
            
            # create pos mask
            pos_mask = self.create_pos_mask(y_true)
            
            # contrastive learning
            l_cl = self.loss(z_all,pos_mask)
            
        else:
            l_cl = None
        
            
        return h, l_cl
class SegmentEncoder(nn.Module):
    def __init__(self,n_poi_groups, nlayers, d_model=128):
        
        super().__init__()

        highway_dim = 6
        gps_dim = 16
        week_dim = 3
        date_dim = 10
        time_dim = 20
        poi_dim = 32
        self.datetime_dim = week_dim + date_dim + time_dim
        
        self.highwayembed = nn.Embedding(17, highway_dim, padding_idx=0)
        self.gpsembed = nn.Linear(4, gps_dim)
        
        self.weekembed = nn.Embedding(8, week_dim)
        self.dateembed = PositionalEncoding1D(date_dim)
        self.timeembed = PositionalEncoding1D(d_model=time_dim)
        
        self.poi_embed = PoiResGatedFilMEncoder(
            n_poi_groups=n_poi_groups,
            time_dim=self.datetime_dim,
            embed_dim=poi_dim,
            n_layers=nlayers
        )
        mlp_in_dim = 2 + 2 *highway_dim + gps_dim + poi_dim # 
        
        
        self.represent = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_in_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(mlp_in_dim * 2, d_model)
        )
        
    def forward(self, links, dateinfo): 
        # links: (B, T, 17) [HighwayID1, HighwayID2, Len, CumLen, Lat1, Lon1, Lat2, Lon2, poi*9]
        # dateinfo: (B, 3)
        # poi_counts: (B, T, n_poi_groups)
        # lens: (B,)
        B, T, _ = links.shape
        
        weekrep   = self.weekembed(dateinfo[:, 0].long())
        daterep   = self.dateembed(dateinfo[:, 1])
        timerep   = self.timeembed(dateinfo[:, 2])
        
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1)
        datetimerep_expand = datetimerep.unsqueeze(1).expand(-1,T, -1) # (B,T,seq_hidden_dim)
        # spatial features
        highwayrep1 = self.highwayembed(links[:, :, 0].long()) # 6
        highwayrep2 = self.highwayembed(links[:, :, 1].long()) # 6
        highwayrep = torch.cat([highwayrep1, highwayrep2], dim=-1)
        
        gpsrep = torch.tanh(self.gpsembed(links[:, :, 4:8].float())) # 16
        
        # poi features
        poirep = self.poi_embed(links[:, :, 8:].float(), datetimerep_expand)
        features = torch.cat([links[..., 2:4], gpsrep,highwayrep, poirep], dim=-1) # 2 + 5 + 16 + 33 
        
        features_proj = self.represent(features) # (B,T,seq_hidden_dim)
        
        return features_proj, datetimerep
    
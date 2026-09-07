from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1,3,1,1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1,3,1,1)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size, attn_hidden=128, mask_strength=0.0, hard_filter_threshold=None):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hidden_size, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, 1))
        self.mask_strength = mask_strength
        self.hard_filter_threshold = hard_filter_threshold
    def forward(self, patch_tokens, mask_prior=None, return_weights=False):
        scores = self.attn(patch_tokens)
        if mask_prior is not None:
            if self.hard_filter_threshold is not None:
                keep = mask_prior >= self.hard_filter_threshold
                empty = keep.sum(dim=1) == 0
                if empty.any():
                    keep = keep.clone(); keep[empty] = True
                neg_inf = torch.finfo(scores.dtype).min
                scores = scores + torch.where(keep.unsqueeze(-1), torch.zeros_like(scores), torch.full_like(scores, neg_inf))
            else:
                log_prior = torch.log(mask_prior.clamp(min=1e-4)).unsqueeze(-1)
                scores = scores + self.mask_strength * log_prior
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * patch_tokens).sum(dim=1)
        return (pooled, weights.squeeze(-1)) if return_weights else pooled

class DeepHead(nn.Module):
    def __init__(self, in_dim, head_dims, dropout=0.3):
        super().__init__()
        layers = [nn.Linear(in_dim, head_dims[0]), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        for i in range(len(head_dims)-1):
            layers.append(nn.Linear(head_dims[i], head_dims[i+1]))
            if i+1 < len(head_dims)-1:
                layers.extend([nn.ReLU(inplace=True), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class TXRVPatchEncoder(nn.Module):
    def __init__(self, input_size=128, pretrained=True):
        super().__init__()
        try:
            import torchxrayvision as xrv
        except ImportError as exc:
            raise ImportError('torchxrayvision is required. Install requirements.txt first.') from exc
        weights = 'densenet121-res224-all' if pretrained else None
        self.model = xrv.models.DenseNet(weights=weights)
        self.input_size = input_size
        self.feature_dim = int(self.model.classifier.in_features)
        self.model.classifier = nn.Identity()
    def forward(self, patches_flat):
        mean = IMAGENET_MEAN.to(patches_flat.device); std = IMAGENET_STD.to(patches_flat.device)
        x = patches_flat * std + mean
        x = x.mean(dim=1, keepdim=True)
        x = x * 2048.0 - 1024.0
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        feats = self.model.features(x)
        feats = F.relu(feats, inplace=False)
        return F.adaptive_avg_pool2d(feats, (1,1)).flatten(1)
    def freeze_all(self):
        for p in self.model.parameters(): p.requires_grad = False
    def unfreeze_all(self):
        for p in self.model.parameters(): p.requires_grad = True
    def unfreeze_last_n_blocks(self, n):
        self.freeze_all()
        if n <= 0: return
        names = [name for name,_ in self.model.features.named_children()]
        blocks = [n for n in names if n.startswith('denseblock')]
        if not blocks:
            self.unfreeze_all(); return
        target = set(blocks[-n:])
        if 'norm5' in names: target.add('norm5')
        for name, module in self.model.features.named_children():
            if name in target:
                for p in module.parameters(): p.requires_grad = True

class PatchMILClassifier(nn.Module):
    def __init__(self, hidden_size=None, attn_hidden=128, head_dims=None, dropout=0.3,
                 pretrained_encoder=True, freeze_encoder=False, context_layers=2, context_heads=8,
                 encoder_type='torchxrayvision', ark_checkpoint_path=None, num_scales=1,
                 txrv_input_size=128, skip_encoder_build=False):
        super().__init__()
        if encoder_type != 'torchxrayvision':
            raise ValueError("Public Patch-MIL build supports encoder_type='torchxrayvision' only.")
        self.num_scales = num_scales
        if skip_encoder_build:
            self.encoder = None
            self.hidden_size = int(hidden_size or 1024)
            self.encoder_blocks = []
        else:
            self.encoder = TXRVPatchEncoder(txrv_input_size, pretrained_encoder)
            self.hidden_size = int(hidden_size or self.encoder.feature_dim)
            self.encoder_blocks = [m for n,m in self.encoder.model.features.named_children() if n.startswith('denseblock')]
            if freeze_encoder: self.encoder.freeze_all()
        self.context_encoder = None
        if context_layers > 0:
            layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=context_heads,
                                               dim_feedforward=self.hidden_size*2, dropout=dropout,
                                               batch_first=True)
            self.context_encoder = nn.TransformerEncoder(layer, num_layers=context_layers)
        self.attention_pool = AttentionPooling(self.hidden_size, attn_hidden)
        dims = list(head_dims or [256,128,1])
        self.head = DeepHead(self.hidden_size, dims, dropout)
        self.instance_classifier = nn.Linear(self.hidden_size, 1)
    def _apply_context(self, tokens, validity):
        if self.context_encoder is not None:
            tokens = self.context_encoder(tokens, src_key_padding_mask=(validity <= 0))
        return tokens
    def _encode_patches(self, patches):
        if self.encoder is None: raise RuntimeError('Encoder was skipped; use forward_from_tokens().')
        b,n,c,h,w = patches.shape
        feats = self.encoder(patches.reshape(b*n,c,h,w))
        return feats.reshape(b,n,self.hidden_size)
    def forward(self, patches, validity, scale_ids=None):
        tokens = self._apply_context(self._encode_patches(patches), validity)
        return self.head(self.attention_pool(tokens, validity)).squeeze(-1)
    def forward_with_attention(self, patches, validity, scale_ids=None):
        tokens = self._apply_context(self._encode_patches(patches), validity)
        pooled, weights = self.attention_pool(tokens, validity, return_weights=True)
        return self.head(pooled).squeeze(-1), weights
    def forward_from_tokens(self, tokens, validity, scale_ids=None):
        tokens = self._apply_context(tokens, validity)
        pooled = self.attention_pool(tokens, validity)
        return self.head(pooled).squeeze(-1)
    def forward_with_attention_from_tokens(self, tokens, validity, scale_ids=None):
        tokens = self._apply_context(tokens, validity)
        pooled, weights = self.attention_pool(tokens, validity, return_weights=True)
        return self.head(pooled).squeeze(-1), weights
    def forward_with_attention_and_instance_from_tokens(self, tokens, validity, scale_ids=None):
        tokens = self._apply_context(tokens, validity)
        pooled, weights = self.attention_pool(tokens, validity, return_weights=True)
        bag_logits = self.head(pooled).squeeze(-1)
        instance_logits = self.instance_classifier(tokens).squeeze(-1)
        critical = instance_logits.masked_fill(validity < 0.5, torch.finfo(instance_logits.dtype).min).max(dim=1).values
        return bag_logits, weights, critical, instance_logits
    def compute_contiguity_loss(self, attn_weights, patch_coords):
        if patch_coords is None or patch_coords.numel() == 0: return attn_weights.new_zeros(())
        centroid = (attn_weights.unsqueeze(-1) * patch_coords).sum(dim=1, keepdim=True)
        sq_dist = ((patch_coords - centroid) ** 2).sum(dim=-1)
        return (attn_weights * sq_dist).sum(dim=1).mean()
    def set_backbone_trainable(self, trainable=True):
        if self.encoder is not None: self.encoder.unfreeze_all() if trainable else self.encoder.freeze_all()
    def set_backbone_partial_trainable(self, n):
        if self.encoder is not None: self.encoder.unfreeze_last_n_blocks(n)
    def get_param_groups(self, lr_backbone, lr_head):
        groups=[]
        if self.encoder is not None:
            p=[x for x in self.encoder.parameters() if x.requires_grad]
            if p: groups.append({'params':p,'lr':lr_backbone})
        p=[x for name,x in self.named_parameters() if not name.startswith('encoder.') and x.requires_grad]
        if p: groups.append({'params':p,'lr':lr_head})
        return groups

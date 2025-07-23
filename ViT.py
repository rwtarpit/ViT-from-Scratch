import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#image = (h,w,c): first preprocess all images and reduce to a constant dimension

class Patch_Embeddings(nn.Module):
    def __init__(self, height: int, width: int, channel : int, patch_size : int, D : int = 1024)-> None:
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.D = D
        self.cls = nn.Parameter(torch.randn(1, 1, D))  # learnable CLS token
        self.patch_size = patch_size #16x16 for ViT large
        self.linear = nn.Linear(patch_size**2*channel,D)
        self.total_patches = (height//patch_size)*(width//patch_size)
        self.pos_embeds = nn.Parameter(torch.randn(1,self.total_patches+1,D)) #(1,num_patches+1,D)
        
    def forward(self,x : torch.Tensor):#(B,C,H,W)
        assert isinstance(x,torch.Tensor), 'img_batch is not a tensor' # img must be tensor
        checks = {
            "Height of image is not correct": x.shape[2] == self.height,
            "Width of image is not correct": x.shape[3] == self.width,
            "Channel of image is not correct": x.shape[1] == self.channel,
        }

        for msg, condition in checks.items():
            assert condition, msg
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)  # [B, C*patch_area, num_patches]
        patches = patches.transpose(1, 2)  # [B, num_patches, patch_dim]
        patch_embeddings =  self.linear(patches)
        cls_tokens = self.cls.expand(x.shape[0],-1,-1) #[B,1,D]
        x = torch.cat([cls_tokens,patch_embeddings],dim=1) #[B,num_patches+1,D]
        x= x + self.pos_embeds #[B,num_patches+1,D]
        return x

class MultiHeadAttention(nn.Module):
    #input: (batch, num_patches+1, D)
    def __init__(self, D: int, heads: int, dropout: float):
        super().__init__()
        self.d_model = D
        self.h = heads
        assert D % heads == 0
        self.d_k = D // heads
        self.w_q = nn.Linear(D, D)
        self.w_k = nn.Linear(D, D)
        self.w_v = nn.Linear(D, D)
        self.w_o = nn.Linear(D, D)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, dropout):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, q, k, v):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, _ = self.attention(q, k, v, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.w_o(x)
        
class LayerNorm(nn.Module):
    def __init__(self, D :int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(D))
        self.bias = nn.Parameter(torch.zeros(D))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class ResidualConnection(nn.Module):
    def __init__(self, D : int, dropout : float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(D)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MLP(nn.Module):
    def __init__(self, D : int, hidden_layer : int, dropout : float)-> None:
        super().__init__()
        self.layer=nn.Sequential(
            nn.Linear(D,hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer,D),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.layer(x)

class EncoderBlock(nn.Module):
    def __init__(self, MSA : MultiHeadAttention, MLP : MLP, dropout : float, D : int )->None:
        super().__init__()
        self.MSA = MSA
        self.norm = nn.ModuleList(LayerNorm(D) for _ in range(2))
        self.residual_conn = nn.ModuleList(ResidualConnection(dropout=dropout,D=D) for _ in range(2))
        self.MLP = MLP
        
    def forward(self, x):
        x = self.residual_conn[0](x, lambda x: self.MSA(self.norm[0](x), self.norm[0](x), self.norm[0](x)))
        x = self.residual_conn[1](x, lambda x: self.MSA(self.norm[0](x), self.norm[0](x), self.norm[0](x)))
        return x
    
class Encoder(nn.Module):
    def __init__(self,encoders : nn.ModuleList)->None:
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
    def forward(self,x):
        for encoder in self.encoders:
            x=encoder(x)
        return x

class MLPHead(nn.Module):
    def __init__(self, num_classes : int, D : int, hidden_layer : int, dropout : float)->None:
        super().__init__()
        # for pre-training
        self.classifier = nn.Sequential(nn.Linear(D,hidden_layer),
                                        #no hidden layer to be used in case of fine tuning
                                        nn.GELU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_layer,num_classes))
                                        #output logits only (for CrossEntropyLoss)
        self.norm = nn.LayerNorm(D)
    def forward(self,x):
        #(batch,cls_token,D)
        x=self.norm(x[:,0,:])
        return self.classifier(x)
    

class ViT(nn.Module):
    def __init__(self,patch_embeddings : Patch_Embeddings, encoder : Encoder, mlphead : MLPHead )->None:
        super().__init__()
        self.patch_embeddings = patch_embeddings
        self.encoder = encoder
        self.mlphead = mlphead
        
    def forward(self,x):
        x=self.patch_embeddings(x)
        x=self.encoder(x)
        x=self.mlphead(x)
        return x
    
def build_ViT(height : int, width : int, channel : int, patch_size: int, D : int, hidden_layer : int, N : int, num_classes : int, dropout : float, heads : int):
    patch_embeddings = Patch_Embeddings(height,width,channel,patch_size,D)
    encoder_list = []
    for i in range(N):
        multi_head_attention = MultiHeadAttention(D,heads,dropout)
        mlp = MLP(D,hidden_layer,dropout)
        encoder_ = EncoderBlock(multi_head_attention,mlp,dropout,D)
        encoder_list.append(encoder_)
    encoder = Encoder(encoder_list)
    mlp_head = MLPHead(num_classes,D,hidden_layer,dropout)
    vit = ViT(patch_embeddings,encoder,mlp_head)
    for p in vit.parameters():
        if p.dim()>1 :
            nn.init.xavier_uniform_(p)
    return vit
        
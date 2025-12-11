import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .helpers import PerceiverResampler
from .vit_3d import ViT


class MyEmbedding(nn.Module):
    def __init__(self, pretrained_visual_encoder=None, pretrained_adapter=None,
                 num_embeddings=32000, embedding_dim=4096, perceiver_num=32,
                 vis_dim=768, patch_size=32, frame_patch_size=4):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.randn((num_embeddings, embedding_dim)), requires_grad=True)
        self.image_token_weight = nn.Parameter(torch.randn((2, embedding_dim)), requires_grad=True)
        self.region_token_weight = nn.Parameter(torch.randn((2, embedding_dim)), requires_grad=True)
        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size


        self.vision_encoder = ViT(
            image_size=512, frames=512, image_patch_size=patch_size, frame_patch_size=frame_patch_size,
            dim=vis_dim, depth=12, heads=8, mlp_dim=2048, dropout=0.1, emb_dropout=0.1
        )


        self.mask_encoder = ViT(
            image_size=256, frames=64, image_patch_size=patch_size, frame_patch_size=16,
            dim=255, depth=3, heads=8, mlp_dim=512, channels=1, dropout=0.1, emb_dropout=0.1
        )


        if pretrained_visual_encoder is not None:
            vit3d_ckpt = torch.load(pretrained_visual_encoder, map_location='cpu')
            self.vision_encoder.load_state_dict(vit3d_ckpt, strict=True)


        for param in self.vision_encoder.parameters():
            param.requires_grad = False


        self.perceiver = PerceiverResampler(dim=vis_dim, num_latents=perceiver_num)
        if pretrained_adapter is not None:
            state_dict = torch.load(pretrained_adapter, map_location='cpu')
            self.perceiver.load_state_dict(state_dict['perceiver'])


        self.fc = nn.Linear(vis_dim, embedding_dim)
        self.mask_fc = nn.Linear(255, embedding_dim)

    def forward(self, vision_x, mask_x, text_input, region2areas):
        B, S, C, H, W, D = next(iter(vision_x.values())).shape


        vision_temp = vision_x['image']
        vision_temp = rearrange(vision_temp, "b S c h w d-> (b S) c h w d")
        vision_temp, _ = self.vision_encoder(vision_temp)
        vision_temp = rearrange(vision_temp, "(b s) v d -> b s v d", b=B, s=S)
        vision_temp = vision_temp.unsqueeze(2)
        vision_temp = self.perceiver(vision_temp)
        n = vision_temp.shape[2]
        vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
        vision_temp = rearrange(vision_temp, "(b T) d -> b T d", b=B, T=n * S)
        image_embedding = vision_temp

        del vision_x['image']


        region_embeddings = vision_x
        mask_embeddings = mask_x

        for key in region_embeddings.keys():
            vision_temp = region_embeddings[key]
            vision_temp = rearrange(vision_temp, "b S c h w d-> (b S) c h w d")
            vision_temp, _ = self.vision_encoder(vision_temp)
            vision_temp = rearrange(vision_temp, "(b s) v d -> b s v d", b=B, s=S)
            vision_temp = vision_temp.unsqueeze(2)
            vision_temp = self.perceiver(vision_temp)
            n = vision_temp.shape[2]
            vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
            vision_temp = rearrange(vision_temp, "(b T) d -> b T d", b=B, T=n * S)
            region_embeddings[key] = vision_temp


            mask_embedding, _ = self.mask_encoder(mask_x[key])
            mask_embedding = torch.mean(mask_embedding, dim=1)
            mask_embeddings[key] = mask_embedding


        image_embedding = self.fc(image_embedding)
        for key in region_embeddings.keys():
            region_embeddings[key] = self.fc(region_embeddings[key])
            mask_embeddings[key] = self.mask_fc(mask_embeddings[key])


        for key in region_embeddings.keys():
            region_embeddings[key] = torch.cat([region_embeddings[key], mask_embeddings[key].unsqueeze(1)], dim=1)


        max_region = len(region_embeddings)
        vision_region_embedding = torch.zeros(
            (B, 33 * max_region, self.embedding_dim), device=text_input.device
        )

        for i in range(B):
            for j in range(len(region2areas[i])):
                region = region2areas[i][j]
                vision_region_embedding[i, j*33:(j+1)*33, :] = region_embeddings[region][i, :, :]


        embedding_weight = torch.cat([self.weight, self.image_token_weight, self.region_token_weight], dim=0)
        embedding_weight = embedding_weight.unsqueeze(0).repeat(B, 1, 1)
        embedding_weight = torch.cat([embedding_weight, image_embedding, vision_region_embedding], dim=1)

        
        text_input = F.one_hot(text_input, embedding_weight.shape[1]).to(vision_region_embedding.dtype)
        out_put = torch.matmul(text_input, embedding_weight)

        return out_put
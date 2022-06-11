import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.models.vlnbert.vlnbert_init import get_vlnbert_models
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

from waypoint_prediction.utils import nms
from vlnce_baselines.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, length2mask)
import math

@baseline_registry.register_policy
class PolicyViewSelectionVLNBERT(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            VLNBERT(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_IDS[config.local_rank]
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class VLNBERT(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions,
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nInitalizing the VLN-BERT model ...')
        self.vln_bert = get_vlnbert_models(config=None)
        self.vln_bert.config.directions = 1  # a trivial number, change during nav
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )

        # Init the RGB encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet152", "TorchVisionResNet50"
        ], "RGB_ENCODER.cnn_type must be TorchVisionResNet152 or TorchVisionResNet50"
        if model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
            self.rgb_encoder = TorchVisionResNet50(
                observation_space,
                model_config.RGB_ENCODER.output_size,
                device,
                spatial_output=model_config.spatial_output,
            )

        # merging visual inputs
        self.space_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(start_dim=2),)
        self.rgb_linear = nn.Sequential(
            nn.Linear(
                model_config.RGB_ENCODER.encode_size,
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Linear(
                model_config.DEPTH_ENCODER.encode_size,
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        self.vismerge_linear = nn.Sequential(
            nn.Linear(
                model_config.DEPTH_ENCODER.output_size + model_config.RGB_ENCODER.output_size + model_config.VISUAL_DIM.directional,
                model_config.VISUAL_DIM.vis_hidden,
            ),
            nn.ReLU(True),
        )

        self.action_state_project = nn.Sequential(
            nn.Linear(model_config.VISUAL_DIM.vis_hidden+model_config.VISUAL_DIM.directional,
            model_config.VISUAL_DIM.vis_hidden),
            nn.Tanh())
        self.action_LayerNorm = BertLayerNorm(
            model_config.VISUAL_DIM.vis_hidden, eps=layer_norm_eps)

        self.drop_env = nn.Dropout(p=0.4)

        self.train()

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(self, mode=None, 
            waypoint_predictor=None,
            observations=None,
            lang_idx_tokens=None, lang_masks=None,
            lang_feats=None, lang_token_type_ids=None,
            headings=None, 
            cand_rgb=None, cand_depth=None,
            cand_direction=None, cand_mask=None,
            masks=None,
            post_states=None, in_train=True):

        if mode == 'language':
            h_t, language_features = self.vln_bert(
                'language', lang_idx_tokens,
                attention_mask=lang_masks, lang_mask=lang_masks,)

            return h_t, language_features

        elif mode == 'waypoint':
            batch_size = observations['instruction'].size(0)
            ''' encoding rgb/depth at all directions ----------------------------- '''
            NUM_ANGLES = 120    # 120 angles 3 degrees each
            NUM_IMGS = 12
            NUM_CLASSES = 12    # 12 distances at each sector
            depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
            rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)

            # reverse the order of input images to clockwise
            # single view images in clockwise agrees with the panoramic image
            a_count = 0
            for i, (k, v) in enumerate(observations.items()):
                if 'depth' in k:
                    for bi in range(v.size(0)):
                        ra_count = (NUM_IMGS - a_count)%NUM_IMGS
                        depth_batch[ra_count+bi*NUM_IMGS] = v[bi]
                        rgb_batch[ra_count+bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                    a_count += 1
            obs_view12 = {}
            obs_view12['depth'] = depth_batch
            obs_view12['rgb'] = rgb_batch
            depth_embedding = self.depth_encoder(obs_view12)
            rgb_embedding = self.rgb_encoder(obs_view12)

            ''' waypoint prediction ----------------------------- '''
            waypoint_heatmap_logits = waypoint_predictor(
                rgb_embedding, depth_embedding)

            # reverse the order of images back to counter-clockwise
            rgb_embed_reshape = rgb_embedding.reshape(
                batch_size, NUM_IMGS, 2048, 7, 7)
            depth_embed_reshape = depth_embedding.reshape(
                batch_size, NUM_IMGS, 128, 4, 4)
            rgb_feats = torch.cat((
                rgb_embed_reshape[:,0:1,:], 
                torch.flip(rgb_embed_reshape[:,1:,:], [1]),
            ), dim=1)
            depth_feats = torch.cat((
                depth_embed_reshape[:,0:1,:], 
                torch.flip(depth_embed_reshape[:,1:,:], [1]),
            ), dim=1)

            # from heatmap to points
            batch_x_norm = torch.softmax(
                waypoint_heatmap_logits.reshape(
                    batch_size, NUM_ANGLES*NUM_CLASSES,
                ), dim=1
            )
            batch_x_norm = batch_x_norm.reshape(
                batch_size, NUM_ANGLES, NUM_CLASSES,
            )
            batch_x_norm_wrap = torch.cat((
                batch_x_norm[:,-1:,:], 
                batch_x_norm, 
                batch_x_norm[:,:1,:]), 
                dim=1)
            batch_output_map = nms(
                batch_x_norm_wrap.unsqueeze(1), 
                max_predictions=5,
                sigma=(7.0,5.0))

            # predicted waypoints before sampling
            batch_output_map = batch_output_map.squeeze(1)[:,1:-1,:]

            candidate_lengths = ((batch_output_map!=0).sum(-1).sum(-1) + 1).tolist()
            if isinstance(candidate_lengths, int):
                candidate_lengths = [candidate_lengths]
            max_candidate = max(candidate_lengths)  # including stop
            cand_mask = length2mask(candidate_lengths, device=self.device)

            if in_train:
                # Augment waypoint prediction
                # parts of heatmap for sampling (fix offset first)
                HEATMAP_OFFSET = 5
                batch_way_heats_regional = torch.cat(
                    (waypoint_heatmap_logits[:,-HEATMAP_OFFSET:,:], 
                    waypoint_heatmap_logits[:,:-HEATMAP_OFFSET,:],
                ), dim=1)
                batch_way_heats_regional = batch_way_heats_regional.reshape(batch_size, 12, 10, 12)
                batch_sample_angle_idxes = []
                batch_sample_distance_idxes = []
                for j in range(batch_size):
                    # angle indexes with candidates
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    # clockwise image indexes (same as batch_x_norm)
                    img_idxes = ((angle_idxes.cpu().numpy()+5) // 10)
                    img_idxes[img_idxes==12] = 0
                    # heatmap regions for sampling
                    way_heats_regional = batch_way_heats_regional[j][img_idxes].view(img_idxes.size, -1)
                    way_heats_probs = F.softmax(way_heats_regional, 1)
                    probs_c = torch.distributions.Categorical(way_heats_probs)
                    way_heats_act = probs_c.sample().detach()
                    sample_angle_idxes = []
                    sample_distance_idxes = []
                    for k, way_act in enumerate(way_heats_act):
                        if img_idxes[k] != 0:
                            angle_pointer = (img_idxes[k] - 1) * 10 + 5
                        else:
                            angle_pointer = 0
                        sample_angle_idxes.append(way_act//12+angle_pointer)
                        sample_distance_idxes.append(way_act%12)
                    batch_sample_angle_idxes.append(sample_angle_idxes)
                    batch_sample_distance_idxes.append(sample_distance_idxes)

            cand_rgb = torch.zeros(
                (batch_size, max_candidate, 2048, 7, 7),
                dtype=torch.float32, device=self.device)
            cand_depth = torch.zeros(
                (batch_size, max_candidate, 128, 4, 4),
                dtype=torch.float32, device=self.device)
            batch_angles = []; batch_angles_c = []
            batch_distances = []
            batch_img_idxes = []
            for j in range(batch_size):
                if in_train:
                    angle_idxes = torch.tensor(batch_sample_angle_idxes[j])
                    distance_idxes = torch.tensor(batch_sample_distance_idxes[j])
                else:
                    # angle indexes with candidates
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    # distance indexes for candidates
                    distance_idxes = batch_output_map[j].nonzero()[:, 1]
                # 2pi- becoz counter-clockwise is the positive direction
                angle_rad_cc = 2*math.pi-angle_idxes.float()/120*2*math.pi
                batch_angles.append(angle_rad_cc.tolist())
                angle_rad_c = angle_idxes.float()/120*2*math.pi
                batch_angles_c.append(angle_rad_c.tolist())
                batch_distances.append(
                    ((distance_idxes + 1)*0.25).tolist())
                # counter-clockwise image indexes
                img_idxes = 12 - (angle_idxes.cpu().numpy()+5) // 10
                img_idxes[img_idxes==12] = 0
                batch_img_idxes.append(img_idxes)
                for k in range(len(img_idxes)):
                    cand_rgb[j][k] = rgb_feats[j][img_idxes[k]]
                    cand_depth[j][k] = depth_feats[j][img_idxes[k]] 
            # use clockwise angles because of vlnbert pretraining
            cand_direction = dir_angle_feature_with_ele(batch_angles_c).to(self.device)
            
            if in_train:
                return cand_rgb, cand_depth, cand_direction, cand_mask, candidate_lengths, batch_angles, batch_distances #, batch_way_log_prob
            else:
                return cand_rgb, cand_depth, cand_direction, cand_mask, candidate_lengths, batch_angles, batch_distances

        elif mode == 'navigation':
            # use clockwise angles because of vlnbert pretraining
            headings = [2*np.pi - k for k in headings]
            prev_actions = angle_feature_with_ele(headings, device=self.device)

            cand_rgb_feats_pool = self.space_pool(cand_rgb)
            cand_rgb_feats_pool = self.drop_env(cand_rgb_feats_pool)
            cand_depth_feats_pool = self.space_pool(cand_depth)

            rgb_in = self.rgb_linear(cand_rgb_feats_pool)
            depth_in = self.depth_linear(cand_depth_feats_pool)

            vis_in = self.vismerge_linear(
                torch.cat((rgb_in, depth_in, cand_direction), dim=2),
            )

            ''' vln-bert processing ------------------------------------- '''
            state_action_embed = torch.cat(
                (lang_feats[:,0,:], prev_actions), dim=1)
            state_with_action = self.action_state_project(state_action_embed)
            state_with_action = self.action_LayerNorm(state_with_action)

            self.vln_bert.config.directions = cand_rgb.size(1)

            state_feats = torch.cat((
                state_with_action.unsqueeze(1), lang_feats[:,1:,:]), dim=1)

            bert_candidate_mask = (cand_mask == 0)
            attention_mask = torch.cat((
                lang_masks, bert_candidate_mask), dim=-1)

            h_t, logits = self.vln_bert('visual',
                state_feats,
                attention_mask=attention_mask,
                lang_mask=lang_masks, vis_mask=bert_candidate_mask,
                img_feats=vis_in)

            return logits, h_t


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

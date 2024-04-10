import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
# from itertools import combinations
from backbone import get_backbone
from utils import *
from torchvision.ops import *


def union_BOX(roi_pers, roi_objs, H=64, W=64):
    assert H == W
    roi_pers = np.array(roi_pers * H, dtype=int)
    roi_objs = np.array(roi_objs * H, dtype=int)
    sample_box = np.zeros([1, 2, H, W])
    
    sample_box[0, 0, roi_pers[1] : roi_pers[3] + 1, roi_pers[0] : roi_pers[2] + 1] = 100    # the first channel is for person
    sample_box[0, 1, roi_objs[1] : roi_objs[3] + 1, roi_objs[0] : roi_objs[2] + 1] = 100    # the second channel is for object
    # the values that are in the locations of people and objects are 100, the rest are 0
    # print("sample_box", sample_box)
    
    return sample_box

def get_attention_maps(normalized_bboxes_per_images):
    persons_np = normalized_bboxes_per_images
    union_box = []
    no_person_dets = len(persons_np)
    for i in range(no_person_dets):
        for j in range(no_person_dets):
            if i!=j:
                union_box.append(union_BOX(persons_np[i], persons_np[j]))
    # print(np.concatenate(union_box).shape)
   
    return np.concatenate(union_box)#shape (N, 2, 64, 64) where N is the number of people * (people-1), 2 is the number of channels, 64 is the height and width


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)

class GCN_Module(nn.Module):
    def __init__(self, cfg):
        super(GCN_Module, self).__init__()
        
        self.cfg=cfg
        
        NFR =cfg.num_features_relation
        
        NG=cfg.num_graph
        N=cfg.num_boxes
        T=cfg.num_frames
        
        NFG=cfg.num_features_boxes
        NFG_ONE=NFG
        
        self.fc_rn_theta_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        self.fc_rn_phi_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        
        
        self.fc_gcn_list=torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(NG) ])
        
        if cfg.dataset_name=='volleyball':
            self.nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([T*N,NFG_ONE]) for i in range(NG) ])
        else:
            self.nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([NFG_ONE]) for i in range(NG) ])


    def forward(self,graph_boxes_features,boxes_in_flat):
        """
        graph_boxes_features  [B*T,N,NFG]
        """
        # GCN graph modeling
        # Prepare boxes similarity relation
        B,N,NFG=graph_boxes_features.shape
        NFR=self.cfg.num_features_relation
        NG=self.cfg.num_graph
        NFG_ONE=NFG
        
        OH, OW=self.cfg.out_size
        pos_threshold=self.cfg.pos_threshold
        
        # Prepare position mask
        graph_boxes_positions=boxes_in_flat  #B*T*N, 4
        graph_boxes_positions[:,0]=(graph_boxes_positions[:,0] + graph_boxes_positions[:,2]) / 2 
        graph_boxes_positions[:,1]=(graph_boxes_positions[:,1] + graph_boxes_positions[:,3]) / 2 
        graph_boxes_positions=graph_boxes_positions[:,:2].reshape(B,N,2)  #B*T, N, 2
        
        graph_boxes_distances=calc_pairwise_distance_3d(graph_boxes_positions,graph_boxes_positions)  #B, N, N
        
        position_mask=( graph_boxes_distances > (pos_threshold*OW) )

        relation_graph=None
        graph_boxes_features_list=[]
        for i in range(NG):
            graph_boxes_features_theta=self.fc_rn_theta_list[i](graph_boxes_features)  #B,N,NFR
            graph_boxes_features_phi=self.fc_rn_phi_list[i](graph_boxes_features)  #B,N,NFR

            similarity_relation_graph=torch.matmul(graph_boxes_features_theta,graph_boxes_features_phi.transpose(1,2))  #B,N,N

            similarity_relation_graph=similarity_relation_graph/np.sqrt(NFR)

            similarity_relation_graph=similarity_relation_graph.reshape(-1,1)  #B*N*N, 1
            # Build relation graph
            relation_graph=similarity_relation_graph

            relation_graph = relation_graph.reshape(B,N,N)

            relation_graph[position_mask]=-float('inf')

            relation_graph = torch.softmax(relation_graph,dim=2) 
              
        
            # Graph convolution
            one_graph_boxes_features=self.fc_gcn_list[i]( torch.matmul(relation_graph,graph_boxes_features) )  #B, N, NFG_ONE
            one_graph_boxes_features=self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features=F.relu(one_graph_boxes_features)
            
            graph_boxes_features_list.append(one_graph_boxes_features)
        
        graph_boxes_features=torch.sum(torch.stack(graph_boxes_features_list),dim=0) #B, N, NFG
        
        return graph_boxes_features,relation_graph

class BasenetFgnnMeanfield(nn.Module):

    def __init__(self, cfg):
        super(BasenetFgnnMeanfield, self).__init__()
        self.cfg=cfg

        D=self.cfg.emb_features # #output feature map channel of backbone
        K=self.cfg.crop_size[0] #crop_size = 5, 5, crop size of roi align
        NFB=self.cfg.num_features_boxes #num_features_boxes = 1024
        OH, OW=self.cfg.out_size    #output size of backbone [50,50]
        num_features_gcn=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, num_features_gcn

        self.backbone = get_backbone(cfg.backbone)  # resnet, vgg16, inception_v3
        
        self.fc_emb_1=nn.Linear(K*K*D,NFB)
        self.nl_emb_1=nn.LayerNorm([NFB])
        
        
        self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])    
        
        
        self.dropout_global=nn.Dropout(p=self.cfg.train_dropout_prob)
    
        self.fc_actions=nn.Linear(NFG,self.cfg.num_actions)
        self.fc_inter_mid=nn.Linear(2*NFG, NFG)
        self.fc_inter_final = nn.Linear(NFG, 2)
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
       
        for p in self.backbone.parameters():
            p.require_grad=False


    def forward(self,batch_data, seq_name, fid, normalized_bboxes):
        images_in, boxes_in, bboxes_num_in = batch_data
        
        
        B=images_in.shape[0]  #batch size
        T=images_in.shape[1]    #num of frames
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size    #output size of backbone [50,50]
        MAX_N=self.cfg.num_boxes    #max num of persons (bboxes) in one picture 15
        NFB=self.cfg.num_features_boxes #num_features_boxes = 1024
        device=images_in.device
        num_features_gcn=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, num_features_gcn

        
        D=self.cfg.emb_features #output feature map channel of backbonem 1056
        K=self.cfg.crop_size[0] #crop_size = 5, 5, crop size of roi align -> 10, 10
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W)) # the image_size is set to [540,960] for all images
                
        # Use backbone to extract features of images_in
        # Pre-process first,normalize the image
        images_in_flat=prep_images(images_in_flat)
        outputs=self.backbone(images_in_flat)   # list of two tensors: torch.Size([2, 288, 65, 117]) and torch.Size([2, 768, 32, 58]) where 2 is the batch size
        
            
    
        # Build multiscale features
        # get the target features before roi align
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)
        boxes_in_flat=torch.reshape(boxes_in,(B*T*MAX_N,4))
        normalized_bboxes=torch.reshape(normalized_bboxes,(B*T,MAX_N,4))    #[batch_size, MAX_N, 4]
            
        boxes_idx=[i * torch.ones(MAX_N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*MAX_N,))

        boxes_idx_flat=boxes_idx_flat.float()
        boxes_idx_flat=torch.reshape(boxes_idx_flat,(-1,1))

        # RoI Align     boxes_features_all：[num_of_person,1056,5,5]
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        normalized_bboxes.requires_grad=False
        
        boxes_features_all=roi_align(features_multiscale,
                                            torch.cat((boxes_idx_flat,boxes_in_flat),1),
                                            (5,5)) 
        
        boxes_features_all=boxes_features_all.reshape(B*T,MAX_N,-1)  #B*T,MAX_N, D*K*K #比如：torch.Size([15, 13, 26400])
        # print(boxes_features_all.size())  #torch.Size([2, 15, 25600]) where 2 is the batch size, 15 is max number of persons in one picture, 26400=1056*5*5
        
        
        # Embedding 
        boxes_features_all=self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all=self.nl_emb_1(boxes_features_all)
        boxes_features_all=F.relu(boxes_features_all)
        
        
        boxes_features_all=boxes_features_all.reshape(B,T,MAX_N,NFB)
        
        boxes_in=boxes_in.reshape(B,T,MAX_N,4)
        
    
        actions_scores=[]
        interaction_scores=[]
        # bboxes_num_in=bboxes_num_in.reshape(B,T)  #B,T,

        bboxes_num_in=bboxes_num_in.reshape(B*T,)
        
        
        for bt in range(B*T):
            # process one frame
            N=bboxes_num_in[bt]
            
            boxes_features=boxes_features_all[bt,:,:N,:].reshape(1,T*N,NFB)
            boxes_positions=boxes_in[bt,:,:N,:].reshape(T*N,4)

            # GCN graph modeling
            for i in range(len(self.gcn_list)):
                graph_boxes_features,relation_graph=self.gcn_list[i](boxes_features,boxes_positions) #1, N, NFG=NFB
    
            # cat graph_boxes_features with boxes_features
            boxes_states=graph_boxes_features+boxes_features  #1, N, NFG
            boxes_states_flat=boxes_states.reshape(-1,NFB)  #1*N, NFB

            boxes_states=self.dropout_global(boxes_states)
            
                    
            # boxes_states=boxes_states.reshape(T,N,NFB)
        
            # Predict actions
            actn_score=self.fc_actions(boxes_states)  #1,N, actn_num
            actn_score=torch.mean(actn_score,dim=0).reshape(N,-1)  #N, actn_num


            # Predict interactions
            interaction_flat=[]
            for i in range(N):
                for j in range(N):
                    if i!=j:
                        interaction_flat.append(torch.cat([boxes_states_flat[i],boxes_states_flat[j]],dim=0))
            interaction_flat=torch.stack(interaction_flat,dim=0) #N(N-1),2048


            interaction_score=self.fc_inter_mid(interaction_flat)  #N(N-1), actn_num
            interaction_score=F.relu(interaction_score)
            interaction_score=self.fc_inter_final(interaction_score)

            
            actions_scores.append(actn_score)
            interaction_scores.append(interaction_score)
            
           


        actions_scores=torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        interaction_scores=torch.cat(interaction_scores,dim=0)
       

        return actions_scores,interaction_scores
        
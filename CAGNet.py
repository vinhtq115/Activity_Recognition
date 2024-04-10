import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np
# from itertools import combinations

from backbone import get_backbone
from utils import *
from torchvision.ops import *

# from meanfield import meanfield
# from mpnn import factor_mpnn
# from utils import unpackEdgeFeature

# def generate_graph(N):
#     """
#     n[0,n-1],z[n,n+n*(n-1)/2-1],vertex node including action label and interaction label
#     h[n+n*(n-1)/2,n+n*(n-1)-1], factor node
#     g[n+n*(n-1),n+n*(n-1)+n*(n-1)*(n-2)/6)-1], factor node
#     build the factor graph
#     :param N: number of node
#     :return: factor graph
#     """

#     G=[[] for _ in range(N+N*(N-1)+N*(N-1)*(N-2)//6)]
#     h_id={ c:i+N for i,c in enumerate(combinations(range(N),2))}#id of z
#     hidx=N+N*(N-1)//2
#     for u in range(N):
#         for v in range(u+1,N):
#             G[hidx].extend([u,v,h_id[(u,v)]])
#             G[u].append(hidx)
#             G[v].append(hidx)
#             G[h_id[(u,v)]].append(hidx)
#             hidx+=1

#     if N>2:
#         gidx=N+N*(N-1)
#         for i in range(N):
#             for j in range(i+1,N):
#                 for k in range(j+1,N):
#                     z1,z2,z3=h_id[(i,j)],h_id[(i,k)],h_id[(j,k)]
#                     G[gidx].extend([z1,z2,z3])
#                     G[z1].append(gidx)
#                     G[z2].append(gidx)
#                     G[z3].append(gidx)
#                     gidx+=1

#     # padding align
#     for l in G:
#         while len(l)<max(3,N-1):
#             l.append(l[-1])
#     return G

# def generate_all_graph(MAXN=15):
#     """
#     generate all graphs
#     :param MAXN:
#     :return:
#     """
#     Gs=[]
#     Gs.extend([[],[]])
#     for n in range(2,MAXN+1):
#         Gs.append(generate_graph(n))
#     return Gs

# def generate_Tri(MAXN=15):
#     """
#     the ordered triangle element cordinates
#     eg. N=3, upper triangle [[0,1],[0,2],[1,2]]
#     the uTri and lTri are used to comnpute the index of z node
#     :param MAXN:
#     :return:
#     """
#     uTri=[]
#     lTri=[]
#     for n in range(0,MAXN+1):
#         if n==0:
#             uTri.append([])
#             lTri.append([])
#             continue
#         utmp=[]
#         ltmp=[]
#         # ordered cordinates
#         for u in range(n):
#             for v in range(u+1,n):
#                 utmp.append([u,v])
#                 ltmp.append([v,u])
#         uTri.append(utmp)
#         lTri.append(ltmp)
#     return uTri,lTri

# def generate_h_cord(MAXN=15):
#     """
#     h_cord is used to retrieve the features in y and z
#     :param MAXN:
#     :return:
#     """
#     h_cord=[]
#     h_cord.extend([[],[]])
#     for N in range(2,MAXN+1):
#         tmp=[[*c,i+N]for i,c in enumerate(combinations(range(N),2))]
#         h_cord.append(tmp)
#     return h_cord

# def generate_g_cord(MAXN=15):
#     """
#     g_cord is used to retrieve the features in z
#     :param MAXN:
#     :return:
#     """
#     g_cord=[]
#     g_cord.extend([[],[],[]])
#     for N in range(3,MAXN+1):
#         h_id={c:i+N for i,c in enumerate(combinations(range(N),2))}  #id of z
#         tmp=[]
#         for i in range(N):
#             for j in range(i+1,N):
#                 for k in range(j+1,N):
#                     z1,z2,z3=h_id[(i,j)],h_id[(i,k)],h_id[(j,k)]
#                     tmp.append([z1,z2,z3])
#         g_cord.append(tmp)
#     return g_cord


# Gs=generate_all_graph()
# uTri,lTri=generate_Tri()
# h_cord=generate_h_cord()
# g_cord=generate_g_cord()

# def get_z_nodefeature(mat,N):
#     """
#     etrieve the upper triangle features, that is z node features,
#     transform feature shape from N*(N-1) to N*(N-1)/2
#     :param mat:
#     :param N:
#     :return:
#     """
#     device=mat.device
#     mat=unpackEdgeFeature(mat,N)
#     uidx=torch.Tensor(uTri[N]).to(device).long()
#     return mat[uidx[:,0],uidx[:,1],:]

# def get_h_factorfeature(nodefeature,N):
#     """
#     node features consist of action feature and interaction feature
#     according to the h_cord
#     the final feature are average of the features of y1,y2,z
#     :param nodefeature:
#     :param N:
#     :return:
#     """
#     device=nodefeature.device
#     h_cord_n=torch.Tensor(h_cord[N]).to(device).long()
#     h_f=nodefeature[h_cord_n]
#     return torch.mean(h_f,dim=1)

# def get_g_factorfeature(nodefeature,N):
#     device=nodefeature.device
#     g_cord_n=torch.Tensor(g_cord[N]).to(device).long()
#     g_f=nodefeature[g_cord_n]
#     return torch.mean(g_f,dim=1)

# def get_edgefeature(nodefeature,factorfeature,N):
#     """
#     compute the edge weight of the factor graph,
#     :param nodefeature: corresponding to the node in fgnn
#     :param factorfeature:corresponding to the factor node in fgnn
#     :param N:
#     :return:
#     """
#     nff=torch.cat((nodefeature,factorfeature),dim=0)# node factor feature
#     device=nodefeature.device
#     graph=torch.Tensor(Gs[N]).to(device).long()
#     ef=torch.cat((nff.unsqueeze(1).repeat((1,max(3,N-1),1)),nff[graph]),dim=-1)# edge feature
#     return ef


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
class BasenetFgnnMeanfield(nn.Module):

    def __init__(self, cfg):
        super(BasenetFgnnMeanfield, self).__init__()
        self.cfg=cfg

        D=self.cfg.emb_features # #output feature map channel of backbone
        K=self.cfg.crop_size[0] #crop_size = 5, 5, crop size of roi align
        NFB=self.cfg.num_features_boxes #num_features_boxes = 1024
        # factor graph para
        NDIM=128# node feature dim
        NMID=64 # node feature mid layer dim
        FDIM=NDIM# factor feature dim
        OH, OW=self.cfg.out_size    #output size of backbone [50,50]

        self.backbone = get_backbone(cfg.backbone)  # resnet, vgg16, inception_v3

        self.fc_emb_1=nn.Linear(K*K*1024,NFB)
        self.dropout_emb_1 = nn.Dropout(p=self.cfg.train_dropout_prob)
        
        # self.fc_action_node=nn.Linear(NFB,NDIM)
        # self.fc_action_mid=nn.Linear(NDIM,NMID)
        # self.nl_action_mid=nn.LayerNorm([NMID])
        # self.fc_action_final=nn.Linear(NMID,self.cfg.num_actions)

        # self.fc_interactions_mid=nn.Linear(2*NFB,NDIM)#
        # self.fc_interactions_final=nn.Linear(NDIM,2)

        # self.fgnn=factor_mpnn(NDIM,[FDIM],
        #                       [64*2,64*2,128*2,128*2,256*2,256*2,128*2,128*2,64*2,64*2,NDIM],
        #                       [16])# fgnn 10 layers


        # self.fc_edge=nn.Linear(2*NDIM,16)

        # self.lambda_h=nn.Parameter(torch.Tensor([self.cfg.lambda_h]))
        # self.lambda_g=nn.Parameter(torch.Tensor([self.cfg.lambda_g]))
        self.flat = Flatten()
        
        self.Conv_people = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=False),
        )
        
        # Context
        self.Conv_context = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=False),
        )
        
        self.lin_visual_head = nn.Sequential(
            # nn.Linear(2048, 29),
            nn.Dropout2d(p=0.5),
            # nn.Linear(, 1024),
            # nn.Linear(lin_size*3, 1024),
            # nn.Linear(lin_size*3+4+sp_size, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.ReLU(),
            #  nn.ReLU(),
        )
        self.lin_visual_tail = nn.Sequential(
            nn.Linear(256, self.cfg.num_actions),
        )

        self.lin_single_head = nn.Sequential(
            # nn.Linear(2048,1),
            nn.Dropout2d(p=0.5),
            nn.Linear(2048, 1024),
            # nn.Linear(lin_size*3, 1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.lin_single_tail = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(512, 2),
            # nn.Linear(10,1),
        )
        self.lin_spmap_tail=nn.Sequential(
                    nn.Linear(512, 2),
                   
                )
        
        ##### Attention Feature Model######
        self.conv_sp_map = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(5, 5)),
            # nn.Conv2d(3, 64, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 32, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.AvgPool2d((13, 13), padding=0, stride=(1, 1)),
            # nn.Linear(32,1024),
            # nn.ReLU()
        )
        self.spmap_up = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(),
        )
        
        # self.fc = nn.Sequential(
        #     nn.Linear(1056*10*10, 528*10*10),
        #     nn.Linear(528*10*10, 264*10*10),
        #     nn.Linear(264*10*10, 1024),
        #     nn.ReLU(),
        # )


        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        for p in self.backbone.parameters():
            p.require_grad=False
        
        # self.sigmoid = nn.Sigmoid()
        # self.fc_interactions_final=nn.Linear(NDIM,2)


    def forward(self,batch_data, seq_name, fid, normalized_bboxes):
        images_in, boxes_in, bboxes_num_in = batch_data
        # print(normalized_bboxes)
        # print(normalized_bboxes.size())
        # print(seq_name) # ('hifive_0028', 'pat_0030')
        # print(fid) # tensor[46, 30]
        # exit()
        # read config parameters
        B=images_in.shape[0]  #batch size
        T=images_in.shape[1]    #num of frames
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size    #output size of backbone [50,50]
        MAX_N=self.cfg.num_boxes    #max num of persons (bboxes) in one picture 15
        NFB=self.cfg.num_features_boxes #num_features_boxes = 1024
        device=images_in.device
        
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
        
        # print('features_multiscale: ', features_multiscale.size()) #torch.Size([2, 1024, 50, 50]) where 2 is the batch size
        
        # get feature for context
        # context_features=[]
        # for features in outputs:
        #     if features.shape[2:4]!=torch.Size([25,25]):
        #         features=F.interpolate(features,size=(25,25),mode='bilinear',align_corners=True)
        #     context_features.append(features)
        # context_features=torch.cat(context_features,dim=1)
        
        
        
        boxes_in_flat=torch.reshape(boxes_in,(B*T*MAX_N,4))
        normalized_bboxes=torch.reshape(normalized_bboxes,(B*T,MAX_N,4))    #[batch_size, MAX_N, 4]
        
        # print('boxes_in_flat: ', boxes_in_flat.size()) #torch.Size([30, 4]) where 30=2*15
        # print(seq_name, fid)
        # print(boxes_in_flat)
        # exit()
            
        boxes_idx=[i * torch.ones(MAX_N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*MAX_N,))

        boxes_idx_flat=boxes_idx_flat.float()
        boxes_idx_flat=torch.reshape(boxes_idx_flat,(-1,1))
        # print('boxes_idx_flat: ', boxes_idx_flat.size()) #torch.Size([30, 1]) where 30=2*15

        # RoI Align     boxes_features_all：[num_of_person,1056,5,5]
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        normalized_bboxes.requires_grad=False
        
        boxes_features_all=roi_align(features_multiscale,
                                            torch.cat((boxes_idx_flat,boxes_in_flat),1),
                                            (5,5)) 
        # print(boxes_features_all.size()) #torch.Size([30, 1024, 10, 10]) where 30=2*15
        # boxes_features_all=boxes_features_all.reshape(B*T,MAX_N,1024, 10, 10)  
        # print(boxes_features_all.size()) #torch.Size([2, 15, 1024, 10, 10]) where 2 is the batch size, 15 is max number of persons in one picture, 1024 is the number of features
        
        boxes_features_all=boxes_features_all.reshape(B*T,MAX_N,-1)  #B*T,MAX_N, D*K*K #比如：torch.Size([15, 13, 26400])
        # print(boxes_features_all.size())  #torch.Size([2, 15, 25600]) where 2 is the batch size, 15 is max number of persons in one picture, 26400=1056*5*5

        # Embedding 
        boxes_features_all=self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
        # print(boxes_features_all.size())
        boxes_features_all=F.relu(boxes_features_all)
        # B*T,MAX_N,NFB
        boxes_features_all=self.dropout_emb_1(boxes_features_all)
        # print(boxes_features_all.size()) #torch.Size([2, 15, 1024]) where 2 is the batch size, 15 is max number of persons in one picture, 1024 is the number of features
        
    
        actions_scores=[]
        interaction_scores=[]

        bboxes_num_in=bboxes_num_in.reshape(B*T,)
        
        ### Defining The Pooling Operations #######
        # pool_size = (10, 10)
        # x, y = features_multiscale.size()[2], features_multiscale.size()[3]
        # hum_pool = nn.AvgPool2d(pool_size, padding=0, stride=(1, 1))
        # context_pool = nn.AvgPool2d((x, y), padding=0, stride=(1, 1))
        
        ### Human###
        # residual_people = boxes_features_all
        # res_people = self.Conv_people(boxes_features_all) + residual_people
        # res_av_people = hum_pool(res_people)
        # out_people = self.flat(res_av_people)
        # print(out2_people.size())  # [N, 1024] where N=2*15=30
        # out_people = out_people.reshape(B*T,MAX_N,NFB)  
        # print(out_people.size())  # [2, 15, 1024] where 2 is the batch size, 15 is max number of persons in one picture, 1024 is the number of features
        
        # Context 
        # residual_context = features_multiscale
        # res_context = self.Conv_context(features_multiscale) 
        # res_av_context = context_pool(res_context)
        # out_context = self.flat(res_av_context)
        # print(out_context.size())  # [N, 1024] where N is the batch size
        
        
        
        for bt in range(B*T):
            # process one frame
            N=bboxes_num_in[bt]
            # uidx=torch.Tensor(uTri[N]).long().to(device)
            # lidx=torch.Tensor(lTri[N]).long().to(device)
            boxes_features=boxes_features_all[bt,:N,:].reshape(1,N,NFB)  #1,N,NFB where N is the number of real persons in one picture and NFB=1024
            normalized_bboxes_per_images = normalized_bboxes[bt,:N,:].reshape(N,4) #[N, 4] where N is the number of real persons in one picture
            
    
            boxes_states=boxes_features  

            NFS=NFB
            # Get spatial attention map
            union_box = get_attention_maps(normalized_bboxes_per_images.detach().cpu().numpy())
            union_box = torch.tensor(union_box).cuda().float()
            # print(union_box.size()) #torch.Size([N, 2, 64, 64]) where N is the number of people * (people-1), 2 is the number of channels, 64 is the height and width
            out_union = self.spmap_up(self.flat(self.conv_sp_map(union_box)))
            # print(out_union.size()) # [N, 512] where N is the number of pairs people * (people-1)
            
            

            # Predict actions
            boxes_states_flat=boxes_states.reshape(-1,NFS)  #1*N, NFS
            lin_visual = self.lin_visual_head(boxes_states_flat)
            actn_scores = self.lin_visual_tail(lin_visual)
            # lin_att=self.lin_spmap_tail(out_union)
            # actn_scores = lin_visual * lin_att
            

            # print(actions_scores.size()) #torch.Size([N, num_classes]) 
            
            
            # actn_score=self.fc_action_node(boxes_states_flat)  #1*N, actn_num
            # actn_score=F.relu(actn_score)
            

            # Predict interactions
            interaction_flat=[]
            for i in range(N):
                for j in range(N):
                    if i!=j:
                        # concatenate features of pairs of humans
                        # interaction_flat.append(torch.cat([boxes_states_flat[i],boxes_states_flat[j], out_context[bt,:]],dim=0))
                        interaction_flat.append(torch.cat([boxes_states_flat[i],boxes_states_flat[j]],dim=0))
            interaction_flat=torch.stack(interaction_flat,dim=0) #N(N-1),2048
            # print(interaction_flat.size()) #torch.Size([N(N-1), 3072])
            lin_single_h = self.lin_single_head(interaction_flat)
            # print(lin_single_h.size()) # [N, 512] where N is the number of pairs people * (people-1)
            lin_single_t = lin_single_h * out_union
            lin_att=self.lin_spmap_tail(out_union)
            
            interaction_score = (self.lin_single_tail(lin_single_t)+lin_att)/2

            # print(interaction_score.size()) # [N, 1] where N is the number of pairs people * (people-1)
            # print(actn_scores.size())
            # print(interaction_score.size())
            actions_scores.append(actn_scores)
            interaction_scores.append(interaction_score)
            # print(actn_scores.size())
            # print(interaction_score.size())

            # # ===== fgnn procedure
            # interaction_flat=self.fc_interactions_mid(interaction_flat) #N*(N-1),num_action
            # interaction_flat=F.relu(interaction_flat)

            # # compute the node feature
            # nodefeature=torch.cat((actn_score,
            #                        get_z_nodefeature(interaction_flat,N)),dim=0)

            # # the fgnn are valid when N>2
            # if N>2:
            #     # compute the factor feature
            #     # if N==2:factorfeature=get_h_factorfeature(nodefeature,N)
            #     # else :factorfeature=torch.cat((get_h_factorfeature(nodefeature,N),
            #     #                          get_g_factorfeature(nodefeature,N)),dim=0)
            #     factorfeature=torch.cat((get_h_factorfeature(nodefeature,N),
            #                              get_g_factorfeature(nodefeature,N)),dim=0)
            #     weight=self.fc_edge(get_edgefeature(nodefeature,factorfeature,N)).unsqueeze(0)
            #     weight=F.relu(weight)
            #     graph=torch.Tensor(Gs[N]).unsqueeze(0).to(device).long()
            #     nodefeature=nodefeature.transpose(0,1).unsqueeze(0).unsqueeze(-1)
            #     factorfeature=factorfeature.transpose(0,1).unsqueeze(0).unsqueeze(-1)

            #     nodefeature,factorfeature=self.fgnn(nodefeature,[factorfeature],[[graph,weight]])

            #     nodefeature=nodefeature.squeeze()
            #     nodefeature=nodefeature.transpose(0,1)
            #     actn_node_score=nodefeature[:N,:]
            #     E=actn_node_score.shape[-1]
            #     interaction_score=torch.zeros((N,N,E)).to(device)
            #     interaction_score[uidx[:,0],uidx[:,1],:]=nodefeature[N:,:]
            #     interaction_score[lidx[:,0],lidx[:,1],:]=nodefeature[N:,:]
            #     interaction_score=packEdgeFeature(interaction_score,N)# N*N=>N*(N-1)

            #     actn_score = self.fc_action_mid(actn_score + actn_node_score)
            #     actn_score=self.nl_action_mid(actn_score)
            #     actn_score=F.relu(actn_score)
            #     actn_score=self.fc_action_final(actn_score)
            #     interaction_score=self.fc_interactions_final(interaction_score)
            # else:
            #     actn_score = self.fc_action_mid(actn_score)
            #     actn_score=self.nl_action_mid(actn_score)
            #     actn_score=F.relu(actn_score)
            #     actn_score=self.fc_action_final(actn_score)
            #     interaction_score=self.fc_interactions_final(interaction_flat) #N(N-1), 2
            # # =====

            # Q_y,Q_z=meanfield(self.cfg, actn_score, interaction_score,
            #                   self.lambda_h,self.lambda_g)
            
            # actions_scores.append(Q_y)
            # interaction_scores.append(Q_z)

        actions_scores=torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        interaction_scores=torch.cat(interaction_scores,dim=0)
        # print(actions_scores.size())
        # print(interaction_scores.size())
        # exit()

        return actions_scores,interaction_scores
        
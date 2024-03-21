import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
class Subspace_Projection(nn.Module):
    def __init__(self, num_dim=5, type = 'cosine', temperature = 0.1):
        super().__init__()
        self.num_dim = num_dim
        self.temperature = temperature
        self.simi_type = type

    def create_subspace(self, supportset_features, class_size, sample_size):
        all_hyper_planes = []
        means = []
        for ii in range(class_size):
            num_sample = sample_size
            all_support_within_class_t = supportset_features[ii]
            all_support_within_class = torch.transpose(all_support_within_class_t, 0, 1)
            uu, s, v = torch.svd(all_support_within_class.double(), some=False)
            uu = uu.float()
            all_hyper_planes.append(uu[:, :self.num_dim])

        all_hyper_planes = torch.stack(all_hyper_planes, dim=0)

        if len(all_hyper_planes.size()) < 3:
            all_hyper_planes = all_hyper_planes.unsqueeze(-1)

        return all_hyper_planes, []


    def projection_metric(self, target_features, hyperplanes, unlabel_features=None):
        eps = 1e-12
        batch_size = len(target_features)
        class_size = hyperplanes.shape[0]

        similarities = []
        unlabel_similarities = []

        discriminative_loss = 0.0
        for j in range(class_size):  ## j-th hyperplane
            one_plane_queryloss = []
            for ind in range(batch_size):  ## index of query image
                target_features[ind] = target_features[ind].float()
                # h_plane_j =  hyperplanes[j].unsqueeze(0)
                h_plane_j =  hyperplanes[j].unsqueeze(0).repeat(target_features[ind].shape[0], 1, 1)
                projected_query_j = torch.bmm(h_plane_j, torch.bmm(torch.transpose(h_plane_j, 1, 2), target_features[ind].unsqueeze(-1)))
                projected_query_dist_inter = target_features[ind] - torch.squeeze(projected_query_j)
                if self.simi_type == 'cosine':
                    cosine_simi = F.cosine_similarity((target_features[ind]).view(target_features[ind].shape[0], 1, -1), 
                                                    (projected_query_j).view(target_features[ind].shape[0], 1, -1), dim=2)
                    # cosine_simi = F.cosine_similarity((target_features[ind]+ common_feat_u_i.expand_as(target_features[ind])).view(target_features[ind].shape[0], 1, -1), 
                                                    # (projected_query_j.squeeze(2)+ common_feat_u_i.expand_as(target_features[ind])).view(target_features[ind].shape[0], 1, -1), dim=2)

                    # query_loss = torch.max(cosine_simi)
                    one_plane_queryloss.append(cosine_simi)
                elif self.simi_type == 'l2_dist':
                    error_dist = -torch.sqrt(torch.sum(projected_query_dist_inter * projected_query_dist_inter
                                                    , dim=-1) + eps)  # norm ||.|| choose the closest dist feature to represent the query img feature loss
                    # query_loss = torch.max(error_dist) 
                    one_plane_queryloss.append(error_dist)
            similarities.append(torch.stack(one_plane_queryloss, dim=0))
            
            ## calculate the loss for subspace seperation
            if class_size > 1:
                for k in range(class_size):
                    if j != k:
                        temp_loss = torch.mm(torch.transpose(hyperplanes[j], 0, 1), hyperplanes[k]) ## discriminative subspaces (Conv4 only, ResNet12 is computationally expensive)
                        discriminative_loss = discriminative_loss + torch.sum(temp_loss*temp_loss)
            else:
                discriminative_loss = 0
        similarities = torch.stack(similarities, dim=1)  ### num_query_img * num_subspace 
        

        if unlabel_features is not None:
            
            for j in range(class_size):  ## j-th hyperplane
                one_plane_queryloss = []
                for ind in range(batch_size):  ## index of query image
                    unlabel_features[ind] = unlabel_features[ind].float()
                    # h_plane_j =  hyperplanes[j].unsqueeze(0)
                    h_plane_j =  hyperplanes[j].unsqueeze(0).repeat(unlabel_features[ind].shape[0], 1, 1)
                    projected_query_j = torch.bmm(h_plane_j, torch.bmm(torch.transpose(h_plane_j, 1, 2), unlabel_features[ind].unsqueeze(-1)))
                    projected_query_dist_inter = unlabel_features[ind] - torch.squeeze(projected_query_j)
                    if self.simi_type == 'cosine':
                        cosine_simi = F.cosine_similarity((unlabel_features[ind]).view(unlabel_features[ind].shape[0], 1, -1), 
                                                        (projected_query_j).view(unlabel_features[ind].shape[0], 1, -1), dim=2)

                        one_plane_queryloss.append(cosine_simi)
                    elif self.simi_type == 'l2_dist':
                        error_dist = -torch.sqrt(torch.sum(projected_query_dist_inter * projected_query_dist_inter
                                                        , dim=-1) + eps)  # norm ||.|| choose the closest dist feature to represent the query img feature loss
                        # query_loss = torch.max(error_dist) 
                        one_plane_queryloss.append(error_dist)
                unlabel_similarities.append(torch.stack(one_plane_queryloss, dim=0))

            unlabel_similarities = torch.stack(unlabel_similarities, dim=1)  ### num_query_img * num_subspace 
            

        return similarities, discriminative_loss, unlabel_similarities

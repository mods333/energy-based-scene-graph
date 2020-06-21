import torch
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import to_onehot, encode_box_info
from maskrcnn_benchmark.modeling.energy_head.graph import Graph

def get_predicted_sg(detections, num_obj_classes):
    '''
    This function converts the detction in scene grpah strucuter 
    '''
    offset = 0
    pair_list = []
    node_list = []
    rel_list = []
    batch_list = []
    edge_batch_list = []

    if 'predict_logits' in detections[0].extra_fields.keys():
        node_key = 'predict_logits'
        is_logits = True
    else:
        node_key = 'pred_labels'
        is_logits = False

    for i, detection in enumerate(detections): 
        pair_list.append(detection.get_field("rel_pair_idxs") + offset)
        node_list.append(detection.get_field(node_key))
        rel_list.append(detection.get_field("pred_rel_scores"))
        batch_list.extend([i]*len(detection)) #For batch wise pooling in the energy model
        edge_batch_list.extend([i]*len(pair_list[-1]))
        offset += len(detection)
    
    node_list = torch.cat(node_list, dim=0)
    if not is_logits:
        node_list = to_onehot(node_list, num_obj_classes, fill=1)
    rel_list = torch.cat(rel_list, dim=0)
    pair_list = torch.cat(pair_list, dim=0)
    batch_list = torch.tensor(batch_list).to(node_list.device)
    edge_batch_list = torch.tensor(edge_batch_list).to(node_list.device)
    # adj_matrix = torch.zeros(size=(node_list.shape[0], node_list.shape[0])).to(node_list.device)
    # adj_matrix[pair_list[:,0], pair_list[:,1]] = 1

    # rel_list = torch.sparse.FloatTensor(pair_list.t(), rel_list, torch.Size([adj_matrix.shape[0], adj_matrix.shape[0], rel_list.shape[-1]])).to_dense()
    
    return node_list, rel_list, pair_list, batch_list, edge_batch_list

def get_gt_scene_graph(targets, num_obj_classes, num_rel_classes):
    '''
    Converts gorund truth annotations into graph structure
    '''
    offset = 0
    pair_list = []
    node_list = []
    rel_list = []
    batch_list = []
    edge_batch_list = []

    for i, target in enumerate(targets):
        rel = target.get_field('relation')
        rel_pair_idxs = torch.nonzero(rel)
        pair_list.append(rel_pair_idxs + offset)
        
        node_list.append(target.get_field('labels'))
        rel_list.append(rel[rel_pair_idxs[:,0], rel_pair_idxs[:,1]])

        batch_list.extend([i]*len(target))
        edge_batch_list.extend([i]*len(pair_list[-1]))
        offset += len(target)

    node_list = to_onehot(torch.cat(node_list, dim=0), num_obj_classes, fill = 1)
    rel_list = to_onehot(torch.cat(rel_list, dim=0), num_rel_classes, fill = 1)
    batch_list = torch.tensor(batch_list).to(node_list.device)
    pair_list = torch.cat(pair_list, dim=0)
    edge_batch_list = torch.tensor(edge_batch_list).to(node_list.device)
    
    # adj_matrix = torch.zeros(size=(node_list.shape[0], node_list.shape[0])).to(node_list.device)
    # adj_matrix[pair_list[:,0], pair_list[:,1]] = 1

    # rel_list = torch.sparse.FloatTensor(pair_list.t(), rel_list, torch.Size([adj_matrix.shape[0], adj_matrix.shape[0], rel_list.shape[-1]])).to_dense()
    
    return node_list, rel_list,  pair_list, batch_list, edge_batch_list

def get_im_graph(images, detections, base_model):
    #Extract region feature from the predicted bounding boxes
    
    features = base_model.backbone(images.tensors)
    node_states = base_model.roi_heads.relation.box_feature_extractor(features, detections)

    return node_states

def detection2graph(images, detections, base_model, num_obj_classes):

    '''
    Create image graph and scene graph given the detections
    Parameters:
    ----------
        images: Batch of input images
        detection: Output of the relation prediction model
        base_model: realtion predcition model (Used of extracting features)
        num_obj_classes: Number of object classes in the dataset(Used for ocnvertin to one hot encoding)
    Return:
    ----------
        im_graph: A graph corresponding to the image
        scene_graph: A graph corresponding to the scene graph
    '''
    #Scene graph Creation
    sg_node_states, sg_rel_states, adj_matrix, batch_list, edge_batch_list = get_predicted_sg(detections, num_obj_classes)

    #Iage graph generation
    im_node_states = get_im_graph(images, detections, base_model)

    scene_graph = Graph(sg_node_states, adj_matrix, batch_list, sg_rel_states, edge_batch_list)
    im_graph = Graph(im_node_states, adj_matrix, batch_list)

    return im_graph, scene_graph, encode_box_info(detections)

def gt2graph(images, targets, base_model, num_obj_classes, num_rel_classes):

    '''
    Create image graph and scene graph given the detections
    Parameters:
    ----------
        images: Batch of input images
        target: Gt Target
        base_model: realtion predcition model (Used of extracting features)
        num_obj_classes: Number of object classes in the dataset(Used for ocnvertin to one hot encoding)
    Return:
    ----------
        im_graph: A graph corresponding to the image
        scene_graph: A graph corresponding to the scene graph
    '''

    sg_node_states, sg_edge_states, adj_matrix, batch_list, edge_batch_list = get_gt_scene_graph(targets, num_obj_classes, num_rel_classes)

    im_node_states = get_im_graph(images, targets, base_model)

    sg_graph = Graph(sg_node_states, adj_matrix, batch_list, sg_edge_states, edge_batch_list)
    im_graph = Graph(im_node_states, adj_matrix, batch_list)
    
    return im_graph, sg_graph, encode_box_info(targets),
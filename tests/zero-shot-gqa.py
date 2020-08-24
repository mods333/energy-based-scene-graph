import json
import os
from tqdm import tqdm
import torch
import ipdb
from maskrcnn_benchmark.utils.miscellaneous import intersect_2d

def main():
    
    num_val_im=5000

    train_sg_path = '/home/ubuntu/SceneGraphEBM/datasets/GQA/sceneGraphs/train_sceneGraphs.json'
    test_sg_path = '/home/ubuntu/SceneGraphEBM/datasets/GQA/sceneGraphs/val_sceneGraphs.json'

    train_image_path = '/home/ubuntu/SceneGraphEBM/datasets/GQA/train_images.json'
    test_image_path = '/home/ubuntu/SceneGraphEBM/datasets/GQA/val_images.json'

    #----------------------------------------------------------------
    print("Loading train and test image ids")
    with open(train_image_path, 'r') as f:
        train_image_ids = json.load(f)
        
    train_image_ids = sorted(list(train_image_ids))

    with open(test_image_path, 'r') as f:
        test_image_ids = json.load(f)
    test_image_ids = sorted(list(test_image_ids))

    #Split into train and vaidation
    val_image_ids = train_image_ids[:num_val_im]
    train_image_ids = train_image_ids[num_val_im:]

    #----------------------------------------------------------------

    print("Loading GQA-train-scene graphs")
    with open(os.path.join(train_sg_path,), 'rb') as f:
        train_sgs = json.load(f)
    print("Loading GQA-test-scene graphs")
    with open(os.path.join(test_sg_path,), 'rb') as f:
        test_sgs = json.load(f)
    
    _,_,_, classes_to_ind, predicates_to_ind, _= load_info(train_sgs, test_sgs)

    print("Splitting sg into train and val")
    #Split train sg into train and val sg
    val_sgs = {}
    val_keys = []
    for key, values in train_sgs.items():
        if key in val_image_ids:
            val_sgs[key] = values
            val_keys.append(key)
        
    for key in val_keys:
        del train_sgs[key]
    #----------------------------------------------------------------
    print("Creating train rel triplets")
    train_triplets = load_graphs(train_sgs, classes_to_ind, predicates_to_ind)
    print("Creating val rel triplets")
    val_triplets = load_graphs(val_sgs, classes_to_ind, predicates_to_ind)
    print("Creating test rel triplets")
    test_triplets = load_graphs(test_sgs, classes_to_ind, predicates_to_ind)

    # train_triplets = torch.tensor(train_triplets)
    # test_triplets = torch.tensor(val_triplets + test_triplets)

    train_triplets = train_triplets
    test_triplets = test_triplets + val_triplets

    zero_shot_triplets = []
    for triplet in tqdm(test_triplets):
        if triplet not in train_triplets:
            zero_shot_triplets.append(triplet)

    zero_shot_triplets = torch.tensor(zero_shot_triplets)
    torch.save(zero_shot_triplets, 'gqa-zeoshot-triplet.pytorch')



def load_graphs(all_sgs_json, labels_to_idx, predicates_to_idx):
    """
    Params:
    ------
        all_sg_json: The dictinaoyt containing the scenegraph informations
    Returns:
    ----------
    A tensror containing tripelet of relations from all the scene graphs
    """

    # Load the image filenames split (i.e. image in train/val/test):
    # train - 0, val - 1, test - 2

    num_graph = len(all_sgs_json)

    rel_triplets = []
    #--------------------------------------------------------------------------------------------------------------------------------------------
    for _, sg_i in all_sgs_json.items():

        sg_objects = sg_i['objects']

        if len(sg_objects) == 0:
            continue
        
        sg_oids = list(sg_objects.keys())
        
        raw_rels = []
        oid_to_obj_class = {}

        #----------------------------------------------------------------
        for oid in sg_oids:
            
            obj = sg_objects[oid]
            oid_to_obj_class[oid] = labels_to_idx[obj['name']]

            for rel in obj['relations']:
                raw_rels.append([oid, rel['object'], rel['name']]) 
        #----------------------------------------------------------------

        rels = []
        for raw_rel_i in raw_rels:
            o1 = oid_to_obj_class[raw_rel_i[0]]
            o2 = oid_to_obj_class[raw_rel_i[1]]
            R = predicates_to_idx[raw_rel_i[2]]
            rels.append([o1, o2, R])
        #----------------------------------------------------------------

        rel_triplets.extend(rels)
    #--------------------------------------------------------------------------------------------------------------------------------------------
    
    return rel_triplets

def load_info(train_sgs, val_sgs):
    """
    Loads the file containing the GQA label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
             classes_to_ind: map from object classes to indices
             predicates_to_ind: map from predicate classes to indices
    """
    info = {'label_to_idx': {}, 'predicate_to_idx': {}, 'attribute_to_idx':{}}

    
    obj_classes = set()
    attribute_classes = set()

    for sg in list(train_sgs.values()) + list(val_sgs.values()):
        for obj in sg['objects'].values():
            obj_classes.add(obj['name'])
            for attr in obj['attributes']:
                attribute_classes.add(attr)
    ind_to_classes = ['__background__'] + sorted(list(obj_classes))
    ind_to_attributes = ['__background__'] + sorted(list(attribute_classes))

    for obj_lbl, name in enumerate(ind_to_classes):
        info['label_to_idx'][name] = obj_lbl

    for attr_lbl, name in enumerate(ind_to_attributes):
        info['attribute_to_idx'][name] = attr_lbl

    rel_classes = set()
    for sg in list(train_sgs.values()) + list(val_sgs.values()):
        for obj in sg['objects'].values():
            for rel in obj['relations']:
                rel_classes.add(rel['name'])
    ind_to_predicates = ['__background__'] + sorted(list(rel_classes))
    for rel_lbl, name in enumerate(ind_to_predicates):
        info['predicate_to_idx'][name] = rel_lbl

    assert info['label_to_idx']['__background__'] == 0
    assert info['predicate_to_idx']['__background__'] == 0

    return (ind_to_classes, ind_to_predicates, ind_to_attributes,
            info['label_to_idx'], info['predicate_to_idx'], info['attribute_to_idx'])



if __name__ == '__main__':
    main()
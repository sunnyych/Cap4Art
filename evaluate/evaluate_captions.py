from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json

def evaluate_captions(annotation_file, result_file):
    """
    Evaluate generated captions against ground truth annotations using COCO metrics.
    
    Args:
        annotation_file (str): Path to the annotation JSON file
        result_file (str): Path to the result JSON file containing generated captions
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Load human reference captions
    coco = COCO(annotation_file)
    
    # Load predicted 
    cocoRes = coco.loadRes(result_file)
    
    # Evalu
    cocoEval = COCOEvalCap(coco, cocoRes)
    
    # Evaluate on a subset of images
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    
    # Evaluate
    cocoEval.evaluate()
    
    # Return results
    return cocoEval.eval 
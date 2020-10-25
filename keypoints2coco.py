import os
import sys
import argparse
from datetime import datetime
import xml.etree.ElementTree as ET

try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level='INFO')
except ImportError:
    import logging
    # Create and configure logger
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger=logging.getLogger()

from utils import write_coco_file

VALID_CLASS_NAME = ['Forklift']
FORKLIFT_CATEGORY_ID = 1

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                    required=True,
                    help='Input path to CVAT xml file to parse')
    parser.add_argument('-o', '--output',
                    required=True,
                    help='Output path for COCO format file')
    parser.add_argument('--dataset',
                    default='COCO Keypoints Dataset',
                    help='Name of the dataset')
    
    args = parser.parse_args()
    return args

def parse_xml_file(xml_path):
    """Read from CVAT xml file. Extract labels, images and annotations 
    from xml tree. Use element's attribute as dict for return data.

    Args:
        xml_path (str): path to cvat xml file

    Returns:
        labels: (list) of available labels
        images: (dict) (keys are image ID) dict of image metadata
        anntations: (dict) (keys are image ID) dict of annotation information
    """
    if not os.path.isfile(xml_path):
        logger.error('No file found at {}'.format(xml_path))
        return {}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # list to hold actual labels
    labels = []
    xml_labels = root.find('.//labels') # return element <labels>
    for label in xml_labels:
        xml_label = label.find('.//name')
        if xml_label is not None:
            labels.append(xml_label.text)

    logger.debug(labels)

    # lists to hold images, annotations
    images = {}
    annotations = {}

    xml_images = root.findall('.//image')
    for xml_image in xml_images:
        # Get image attributes dict: id, name, width, height
        image_attrib = xml_image.attrib
        image_id = int(image_attrib['id'])
        images[image_id] = image_attrib

        # Get keypoints: each forklift has 1 <points> element
        # it has attrib 'points' containing list of keypoints separated
        # by semicolon ';'
        # See CVAT XML keypoints format
        annotations[image_id] = {
            'bboxes': [],
            'keypoints': []
        }
        for item in xml_image:
            if item.tag == 'box':
                annotations[image_id]['bboxes'].append(item.attrib)
            elif item.tag == 'points':
                annotations[image_id]['keypoints'].append(item.attrib)

    logger.debug(images)
    logger.debug(annotations)

    return labels, images, annotations

def construct_coco_keypoints(dataset_name, labels, images, annotations):
    """Construct COCO format data from cvat data annotations.
    Convert CVAT points annotation data to COCO annotation data.
    See: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

    Args:
        labels (list): List of available categories for keypoint annotations
        images (dict): Dictionary of image attributes comes from cvat xml. 
                        Included fields: id, name, width, height.
        annotations (dict): Dictionary of points attributes comes from cvat xml.
                            Included fields are label, occluded, points.
    
    Returns:
        coco (dict): full COCO format keypoints data
    """
    now = datetime.now()
    coco = {
        'info': {
            'description': 'COCO 2017 Dataset',
            'url': '',
            'version': '1.0',
            'year': now.strftime('%Y'),
            'contributor': 'COCO Consortium',
            'date_created': now.strftime('%m/%d/%Y')
        },
        'categories': [],
        'licenses': '',
        'images': [],
        'annotations': []
    }

    # Create category list for COCO
    coco_categories = []
    keypoint_labels = [
        'LFrontWheel', 'RFrontWheel', 'LBackWheel', 'RBackWheel',
        'LBlade', 'RBlade',
        'LFrontRoof', 'RFrontRoof', 'LBackRoof', 'RBackRoof']
    for i, cat in enumerate(labels):
        if cat not in VALID_CLASS_NAME:
            continue
        category = {
            'id': i,
            'name': cat,
            'keypoints': keypoint_labels,
            'skeleton': []
        }
        coco_categories.append(category)
    coco['categories'] = coco_categories
    logger.debug('Categories: {}'.format(coco_categories))

    image_ids = images.keys()
    # Transform image dict to coco list
    coco_images = []
    for idx in image_ids:
        item = images[idx]
        coco_image = {
            'file_name': item['name'],
            'height': item['height'],
            'width': item['width'],
            'id': int(idx)
        }
        coco_images.append(coco_image)
    coco['images'] = coco_images

    # Transform annotation dict to coco dict
    # See COCO visibility code: https://github.com/cocodataset/cocoapi/issues/130#issuecomment-369147997
    # 0: not in image
    # 1: in image but hidden
    # 2: clearly visible
    coco_annotations = []
    bboxes = []
    annotation_index = 0
    for idx in image_ids:
        # Firstly, append all the boxes to annotations
        xml_boxes = annotations[idx]['bboxes']
        for xml_box in xml_boxes:
            annotation_index += 1
            coco_box = {
                "segmentation": [],
                "num_keypoints": 10,
                "area": 0.0,
                "iscrowd": 0,
                "keypoints": [],
                "image_id": idx,
                "bbox": [],
                "category_id": FORKLIFT_CATEGORY_ID,
                "id": annotation_index
            }
            label = xml_box['label']
            x1 = float(xml_box['xtl'])
            y1 = float(xml_box['ytl'])
            x2 = float(xml_box['xbr'])
            y2 = float(xml_box['ybr'])
            coco_box['bbox'] = [x1, y1, x2-x1, y2-y1]

            # Create an ordered list to hold keypoints for each bbox
            # Each keypoint occupied 3 slots in array: x, y, visibility
            keypoints = [0] * 30

            # Then we read keypoints and assign them to the bboxes
            xml_keypoints = annotations[idx]['keypoints']
            for xml_keypoint in xml_keypoints:
                p_label = xml_keypoint['label']
                logger.debug(p_label)
                p_visibility = 1 if xml_keypoint['occluded'] == '1' else 2
                p = eval('(' + xml_keypoint['points'] + ')')

                keypoint_index = keypoint_labels.index(p_label) * 3
                if is_inside_box(p, [x1, y1, x2, y2]):
                    keypoints[keypoint_index:keypoint_index+3] = [p[0], p[1], p_visibility]

            coco_box['keypoints'] = keypoints
            coco_annotations.append(coco_box)

        coco['annotations'] = coco_annotations

    return coco

def is_inside_box(point, box):
    if point[0] >= box[0] \
        and point[0] <= box[2] \
        and point[1] >= box[1] \
        and point[1] <= box[3]:
        return True
    return False


if __name__ == '__main__':
    args = parse_args()
    xml_path = args.input
    coco_path = args.output
    dataset_name = args.dataset

    labels, images, annotations = parse_xml_file(xml_path)
    coco = construct_coco_keypoints(dataset_name, labels, images, annotations)
    write_coco_file(coco, coco_path)
    
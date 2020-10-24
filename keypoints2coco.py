import os
import argparse
from datetime import datetime
import xml.etree.ElementTree as ET

try:
    from loguru import logger
except ImportError:
    import logging
    # Create and configure logger
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger=logging.getLogger()

from utils import write_coco_file

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
        annotations[image_id] = []
        xml_keypoints = xml_image.find('.//points')
        if xml_keypoints is not None:
            annotations[image_id].append(xml_keypoints.attrib)

    logger.debug(images)
    logger.debug(annotations)

    return labels, images, annotations

def construct_coco_keypoints(dataset_name, labels, images, annotations):
    """Construct COCO format data from cvat data annotations.
    Convert CVAT points annotation data to COCO annotation data.

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
        'licenses': '',
        'images': [],
        'annotations': []
    }

    image_ids = images.keys()
    # Transform image dict to coco list
    coco_images = []
    for idx in image_ids:
        item = images[idx]

    return coco



if __name__ == '__main__':
    args = parse_args()
    xml_path = args.input
    coco_path = args.output
    dataset_name = args.dataset

    labels, images, annotations = parse_xml_file(xml_path)
    coco = construct_coco_keypoints(dataset_name, labels, images, annotations)
    write_coco_file(coco, coco_path)
    
import os
import json

try:
    from loguru import logger
except ImportError:
    import logging
    # Create and configure logger
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger=logging.getLogger()

def write_coco_file(data, path):
    """Write COCO data to file

    Args:
        data (dict): COCO content as dict
        path (str): Path to write data
    """
    with open(path, 'w') as f:
        json.dump(data, f)
    logger.info('COCO data written to {}'.format(path))

import os
import numpy as np
import scipy.io as sio
import shutil
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString


def make_voc_dir():
    # labels 目录若不存在，创建labels目录。若存在，则清空目录
    if not os.path.exists('../VOC2007/Annotations'):
        os.makedirs('../VOC2007/Annotations')
    if not os.path.exists('../VOC2007/ImageSets'):
        os.makedirs('../VOC2007/ImageSets')
        os.makedirs('../VOC2007/ImageSets/Main')
    if not os.path.exists('../VOC2007/JPEGImages'):
        os.makedirs('../VOC2007/JPEGImages')


def process_annotations(root_dir, save_dir):
    """Process all annotation MAT files"""

    annotation_dir = os.path.join(root_dir, 'annotations')
    file_names = sorted(os.listdir(annotation_dir))

    train_imnames = sio.loadmat(os.path.join(root_dir, 'frame_train.mat'))
    train_imnames = train_imnames['img_index_train'].squeeze()
    test_imnames = sio.loadmat(os.path.join(root_dir, 'frame_test.mat'))
    test_imnames = test_imnames['img_index_test'].squeeze()
    train_imnames = [train_name[0] + '.jpg' for train_name in train_imnames]
    test_imnames = [test_name[0] + '.jpg' for test_name in test_imnames]

    train_box_imnames = []
    train_boxes = []
    test_box_imnames = []
    test_boxes = []

    for i, f_name in enumerate(file_names, 1):
        im_name = f_name[:-4]  # 'c1s1_000151.jpg'
        f_dir = os.path.join(annotation_dir, f_name)  # '../PRW\\annotations\\c1s1_000151.jpg.mat'
        boxes = sio.loadmat(f_dir)
        if 'box_new' in boxes.keys():
            boxes = boxes['box_new']
        elif 'anno_file' in boxes.keys():
            boxes = boxes['anno_file']
        elif 'anno_previous' in boxes.keys():
            boxes = boxes['anno_previous']
        else:
            raise KeyError(boxes.keys())
        valid_index = np.where((boxes[:, 3] > 0) & (boxes[:, 4] > 0))[0]
        assert valid_index.size > 0, \
            'Warning: {} has no valid boxes.'.format(im_name)
        boxes = boxes[valid_index].astype(np.int32)  # <class 'tuple'>: (2, 5)

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'JPEGImages'
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = 'VOC2007/JPEGImages/%s' % im_name
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = '%s' % 640
        node_height = SubElement(node_size, 'height')
        node_height.text = '%s' % 480
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'

        for box_idx, box in enumerate(boxes):
            xmin = box[1] + 1
            ymin = box[2] + 1
            obj_width = box[3]
            obj_height = box[4]
            xmax = xmin + obj_width
            ymax = ymin + obj_height
            difficult = 0

            if obj_height <= 4 or obj_width <= 4:
                difficult = 1

            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = 'person'
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '%s' % difficult
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = '%s' % xmin
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = '%s' % ymin
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = '%s' % xmax
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = '%s' % ymax
            node_name = SubElement(node_object, 'pose')
            node_name.text = 'Unspecified'
            node_name = SubElement(node_object, 'truncated')
            node_name.text = '0'

        image_path = VOCRoot + '/JPEGImages/' + im_name
        xml = tostring(node_root, pretty_print=True)  # 'annotation'
        dom = parseString(xml)
        xml_name = im_name.replace('.jpg', '.xml')
        xml_path = VOCRoot + '/Annotations/' + xml_name
        with open(xml_path, 'wb') as f:
            f.write(xml)


if __name__ == '__main__':
    root_dir = '../../data/person_search/PRW'
    VOCRoot = '../../data/person_search/PRW_VOC'
    # make_voc_dir()

    print('Processing the mat files...')
    process_annotations(root_dir, VOCRoot)
    print('Dataset processing done.')
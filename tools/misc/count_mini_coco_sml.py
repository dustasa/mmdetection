from pycocotools.coco import COCO

# 文件路径
dataDir = '/home/aosun/Desktop/code/mmdetection/data/mini_coco/annotations'
dataType = 'train2017'
annFile = '{}/mini_instances_{}.json'.format(dataDir, dataType)

if __name__ == '__main__':

    # initialize COCO api for instance annotations
    coco_train = COCO(annFile)

    # display COCO categories and supercategories
    # 显示所有类别
    cats = coco_train.loadCats(coco_train.getCatIds())
    cat_nms = [cat['name'] for cat in cats]
    print('COCO 类别:\n{}'.format('\n'.join(cat_nms)) + '\n')

    # 统计单个类别的图片数量与标注数量
    for cat_name in cat_nms:
        catId = coco_train.getCatIds(catNms=cat_name)
        print(catId)
        imgId = coco_train.getImgIds(catIds=catId)
        annId = coco_train.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=False)
        print("{:<15} {:<6d}     {:<10d}\n".format(cat_name, len(imgId), len(annId)))

        # if cat_name == "person":
        #     print(catId)
        #     imgId = coco_train.getImgIds(catIds=catId)
        #     annId = coco_train.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=False)
        #     print("{:<15} {:<6d}     {:<10d}\n".format(cat_name, len(imgId), len(annId)))

    # 统计全部的类别及全部的图片数量和标注数量
    print("全部类别数: " + str(len(coco_train.dataset['categories'])))
    print("全部图片数量: " + str(len(coco_train.dataset['images'])))
    print("全部标注数量: " + str(len(coco_train.dataset['annotations'])))
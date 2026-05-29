import dota_utils as util
import os
import cv2
import json

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
'''
针对DOTA数据集而言，困难度为2的数据需要考虑设计成背景样本还是忽略样本，
'''
def DOTA2COCO(srcpath, destfile,difficult,trainval=True):   ### 可以添加一个difficult参数，我们可以过滤掉困难度为2的图形，
    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(wordname_15):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)
            
            if trainval:
                # annotations
                objects = util.parse_dota_poly2(file)
                for obj in objects:

                    # if obj['difficult'] == difficult:
                    #     print('difficult: ', difficult)
                    #     continue
                    single_obj = {}
                    single_obj['area'] = obj['area']
                    single_obj['category_id'] = wordname_15.index(obj['name']) + 1
                    single_obj['segmentation'] = []
                    single_obj['segmentation'].append(obj['poly'])
                    single_obj['iscrowd'] = 0
                    ### 根据需要的GT坐标的格式进行不同的调整。
                    xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                            max(obj['poly'][0::2]), max(obj['poly'][1::2])

                    width, height = xmax - xmin, ymax - ymin
                    single_obj['bbox'] = xmin, ymin, width, height
            
                    # x1 = obj['poly'][0]
                    # y1 = obj['poly'][1]
                    # x2 = obj['poly'][2]
                    # y2 = obj['poly'][3]
                    # x3 = obj['poly'][4]
                    # y3 = obj['poly'][5]
                    # x4 = obj['poly'][6]
                    # y4 = obj['poly'][7]
                    # single_obj['bbox'] = x1, y1, x2, y2, x3, y3, x4, y4

                    single_obj['image_id'] = image_id
                    data_dict['annotations'].append(single_obj)
                    single_obj['id'] = inst_count
                    inst_count = inst_count + 1

            image_id = image_id + 1
        json.dump(data_dict, f_out)
if __name__ == '__main__':
    DOTA2COCO(r'/data0/data_dj/1024_new', r'/data0/data_dj/1024_new/DOTA_trainval1024.json',difficult=2,trainval=True)

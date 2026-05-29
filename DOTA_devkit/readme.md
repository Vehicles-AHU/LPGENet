
## Update


<p>
    We now add the cpu multi process version of "ImgSplit" and "ResultMerge", gpu version of polygon nms.
</p>

## Functions

The code is useful for <a href="http://captain.whu.edu.cn/DOTAweb/">DOTA<a> or
<a href="http://captain.whu.edu.cn/ODAI/">ODAI<a>. The code provide the following function
<ul>
    <li>
        Load and image, and show the bounding box on it.
    </li>
    <li>
        Evaluate the result.
    </li>
    <li>
        Split and merge the picture and label.
    </li>
</ul>

### What is DOTA?
<p>
Dota is a large-scale dataset for object detection in aerial images. 
It can be used to develop and evaluate object detectors in aerial images. 
We will continue to update DOTA, to grow in size and scope and to reflect evolving real-world conditions.
Different from general object detectin dataset. Each instance of DOTA is labeled by an arbitrary (8 d.o.f.) quadrilateral.
For the detail of <strong style="color:blue"> DOTA-v1.0</strong>, you can refer to our 
<a href="https://arxiv.org/abs/1711.10398">paper</a>.
</p>

### What is DOAI?

[DOAI2019](https://captain-whu.github.io/DOAI2019) is a contest of Detecting Objects in Aerial Images on [CVPR'2019]("http://cvpr2019.thecvf.com/"). It is based on DOTA-v1.5.



[DOAI2018](https://captain-whu.github.io/ODAI) is a contest of object detetion in aerial images on [ICPR'2018]("http://www.icpr2018.org/"). It is based on [DOTA-v1]("http://captain.whu.edu.cn/DOTAweb/"). The contest is closed now. 


### Installation
1. install swig
```
    sudo apt-get install swig
```
2. create the c++ extension for python
```
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
```

### Usage
1. Reading and visualizing data, you can use DOTA.py
2. Evaluating the result, you can refer to the "dota_evaluation_task1.py" and "dota_evaluation_task2.py" (or "dota-v1.5_evaluation_task1.py" and "dota-v1.5_evaluation_task2.py" for DOTA-v1.5)
3. Split the large image, you can refer to the "ImgSplit"
4. Merging the results detected on the patches, you can refer to the ResultMerge.py

An example is shown in the demo.
The subdirectory of "basepath"(which is used in "DOTA.py", "ImgSplit.py") is in the structure of
```
.
├── images
└── labelTxt
```
### 文件说明和操作
1. 针对训练集数据使用"ImgSplit_multi_process.py" or "ImgSplit.py"进行操作
2. 针对测试集数据使用"SplitOnlyImage_multi_process.py"or "SplitOnlyImage.py" 进行操作
3. 格式转换：
  - 针对不同的应用场景，主要转换格式包括mmdet(.pkl),coco格式，json格式，YOLO格式等
  - mmdet: "convert_dota_to_mmdet.py"
  - coco: "DOTA2COCO.py"
  - json: "DOTA2JSON.py"
  - yolo: "dota2yolo.py" ,此处的YOLO为yolov5格式
4. 最后，可能有的文件还需要生成txt文件，通过"create_txt.py"生成

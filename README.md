# BinSegTrain
## Usage
### Before Training
- for adding copy-paste data, please prepare templates (labelme format) and backgrounds, and put them into './datasets/YOUR_DATASET_NAME/tm' and './datasets/YOUR_DATASET_NAME/bg', respectively.  (see examples with YOUR_DATASET_NAME=01)
- prepare the object CAD model in .obj format (see example with 'part01.obj')
- prepare the virtual data output from CaoRui

#### transform virtual dataset output to detectron2 format for training segmentation network
```bash
python datasets/transform_from_virtual.py --obj_idx YOUR_DATASET_NAME --virtual_dir DIRECTORY_OF_VIRTUAL_DATA --obj_path PATH_OF_CAD_MODEL --oc VISIBILITY_THRESHOLD
````
for example:
```bash
python datasets/transform_from_virtual.py --obj_idx 01 --virtual_dir '../example_virtual_data_output/' --obj_path './part01.obj' --oc 0.8
````
you should get folder 'train' containing training images and 'json' containing train.json for annotations, the position is in YOUR_DATASET_NAME, which equals to 'obj_idx' for convinience

#### add copy paste data by 
```bash
python datasets/add_copy_paste.py --name YOUR_DATASET_NAME --gen_num_per_base NUMBER_OF_OUTPUT_IMAGES_FOR_EACH_BACKGROUND_IMAGE  --oc VISIBILITY_THRESHOLD
````
for example:
```bash
python datasets/add_copy_paste.py --name 01 --gen_num_per_base 100  --oc 0.8
````
this will increase the images in 'train' folder and update the train.json file.

#### have a check of the generated data and annotations
```bash
python vis_dataset.py --name YOUR_DATASET_NAME 
````
### Train
Please modify the output path in user config file.
```bash
python train-rmRCNN.py --data_dir YOUR_DATASET_PATH --user_config_file YOUR_CONFIG_PATH 
````
for example:
```bash
python train-rmRCNN.py --data_dir './datasets/01' --user_config_file "./configs/user_rmRCNN.yaml" 
````

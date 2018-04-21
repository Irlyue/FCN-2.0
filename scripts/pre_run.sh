# change the environment variable to load module dynamically
export PYTHONPATH=$PYTHONPATH:/home/wenfeng/models/models/research/slim

tar -zxf ~/models/weights/resnet_v1_50_2016_08_28.tar.gz -C /tmp/
mv /tmp/resnet_v1_50.ckpt /tmp/model_to_restore.ckpt

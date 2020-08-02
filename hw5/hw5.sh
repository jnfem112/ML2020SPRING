#!/bin/bash

wget https://www.dropbox.com/s/d7t9lqnastp98x2/CNN.pkl?dl=0 -O CNN.pkl

python3 saliency_map.py --image_directory=$1 --image_name=0_28.jpg --output_directory=$2 --output_file=saliency_map_1.png
python3 saliency_map.py --image_directory=$1 --image_name=1_2.jpg --output_directory=$2 --output_file=saliency_map_2.png
python3 saliency_map.py --image_directory=$1 --image_name=2_14.jpg --output_directory=$2 --output_file=saliency_map_3.png
python3 saliency_map.py --image_directory=$1 --image_name=3_0.jpg --output_directory=$2 --output_file=saliency_map_4.png
python3 saliency_map.py --image_directory=$1 --image_name=4_7.jpg --output_directory=$2 --output_file=saliency_map_5.png
python3 saliency_map.py --image_directory=$1 --image_name=5_5.jpg --output_directory=$2 --output_file=saliency_map_6.png
python3 saliency_map.py --image_directory=$1 --image_name=6_3.jpg --output_directory=$2 --output_file=saliency_map_7.png
python3 saliency_map.py --image_directory=$1 --image_name=7_3.jpg --output_directory=$2 --output_file=saliency_map_8.png
python3 saliency_map.py --image_directory=$1 --image_name=8_59.jpg --output_directory=$2 --output_file=saliency_map_9.png
python3 saliency_map.py --image_directory=$1 --image_name=9_16.jpg --output_directory=$2 --output_file=saliency_map_10.png
python3 saliency_map.py --image_directory=$1 --image_name=10_5.jpg --output_directory=$2 --output_file=saliency_map_11.png

python3 filter_visualization.py --layer=3 --filter=1 --output_directory=$2 --output_file=filter_visualization_1.png
python3 filter_visualization.py --layer=3 --filter=3 --output_directory=$2 --output_file=filter_visualization_2.png
python3 filter_visualization.py --layer=3 --filter=7 --output_directory=$2 --output_file=filter_visualization_3.png
python3 filter_visualization.py --layer=3 --filter=12 --output_directory=$2 --output_file=filter_visualization_4.png
python3 filter_visualization.py --layer=11 --filter=1 --output_directory=$2 --output_file=filter_visualization_5.png
python3 filter_visualization.py --layer=11 --filter=3 --output_directory=$2 --output_file=filter_visualization_6.png
python3 filter_visualization.py --layer=11 --filter=4 --output_directory=$2 --output_file=filter_visualization_7.png
python3 filter_visualization.py --layer=11 --filter=7 --output_directory=$2 --output_file=filter_visualization_8.png
python3 filter_visualization.py --layer=19 --filter=0 --output_directory=$2 --output_file=filter_visualization_9.png
python3 filter_visualization.py --layer=19 --filter=5 --output_directory=$2 --output_file=filter_visualization_10.png
python3 filter_visualization.py --layer=19 --filter=6 --output_directory=$2 --output_file=filter_visualization_11.png
python3 filter_visualization.py --layer=19 --filter=18 --output_directory=$2 --output_file=filter_visualization_12.png
python3 filter_visualization.py --layer=27 --filter=1 --output_directory=$2 --output_file=filter_visualization_13.png
python3 filter_visualization.py --layer=27 --filter=4 --output_directory=$2 --output_file=filter_visualization_14.png
python3 filter_visualization.py --layer=27 --filter=11 --output_directory=$2 --output_file=filter_visualization_15.png
python3 filter_visualization.py --layer=27 --filter=13 --output_directory=$2 --output_file=filter_visualization_16.png

python3 activation_map.py --image_directory=$1 --image_name=0_0.jpg --layer=3 --filter=1 --output_directory=$2 --output_file=activation_map_1.png
python3 activation_map.py --image_directory=$1 --image_name=0_0.jpg --layer=11 --filter=4 --output_directory=$2 --output_file=activation_map_2.png
python3 activation_map.py --image_directory=$1 --image_name=0_0.jpg --layer=19 --filter=18 --output_directory=$2 --output_file=activation_map_3.png
python3 activation_map.py --image_directory=$1 --image_name=0_0.jpg --layer=27 --filter=4 --output_directory=$2 --output_file=activation_map_4.png

python3 LIME.py --image_directory=$1 --image_name=0_28.jpg --output_directory=$2 -output_file=LIME_1.png
python3 LIME.py --image_directory=$1 --image_name=1_2.jpg --output_directory=$2 -output_file=LIME_2.png
python3 LIME.py --image_directory=$1 --image_name=2_14.jpg --output_directory=$2 -output_file=LIME_3.png
python3 LIME.py --image_directory=$1 --image_name=3_0.jpg --output_directory=$2 -output_file=LIME_4.png
python3 LIME.py --image_directory=$1 --image_name=4_7.jpg --output_directory=$2 -output_file=LIME_5.png
python3 LIME.py --image_directory=$1 --image_name=5_5.jpg --output_directory=$2 -output_file=LIME_6.png
python3 LIME.py --image_directory=$1 --image_name=6_3.jpg --output_directory=$2 -output_file=LIME_7.png
python3 LIME.py --image_directory=$1 --image_name=7_3.jpg --output_directory=$2 -output_file=LIME_8.png
python3 LIME.py --image_directory=$1 --image_name=8_59.jpg --output_directory=$2 -output_file=LIME_9.png
python3 LIME.py --image_directory=$1 --image_name=9_16.jpg --output_directory=$2 -output_file=LIME_10.png
python3 LIME.py --image_directory=$1 --image_name=10_5.jpg --output_directory=$2 -output_file=LIME_11.png

python3 deep_dream.py --image_directory=$1 --image_name=0_0.jpg --layer=27 --learning_rate=0.005 --epoch=100 --output_directory=$2 -output_file=deep_dream_1.png
python3 deep_dream.py --image_directory=$1 --image_name=2_0.jpg --layer=27 --learning_rate=0.005 --epoch=100 --output_directory=$2 -output_file=deep_dream_2.png
python3 deep_dream.py --image_directory=$1 --image_name=4_0.jpg --layer=27 --learning_rate=0.005 --epoch=100 --output_directory=$2 -output_file=deep_dream_3.png
#!/bin/bash
source ~/anaconda2/etc/profile.d/conda.sh
conda activate tf_2_p_3
python pix2pix_1.py --inpath "./train_input" --outpath './train_desired_output' --path_output "./output_model/" --predpath './image_to_pred_landscape' --pred_output "./image_predicted_landscape/" --flag_train 'False' --epochs 800 --checkpoint_dir './checkpoints_landscape/ckpt-21' --restore 'True' --evolution_images 10 --batch_size_train 10 --batch_size_test 1 --img_width 512 --img_height 512 --img_aug 40 --output_channels 3 --_lambda 100 
conda deactivate

conda activate tf_1_14_p_2_7
python2.7 enhancenet.py --path_input './image_predicted_landscape' --path_output './enhance_image_landscape'
conda deactivate


source ~/anaconda2/etc/profile.d/conda.sh
conda activate tf_2_p_3
python pix2pix_1.py --inpath "./train_input" --outpath './train_desired_output' --path_output "./output_model/" --predpath './image_to_pred_car' --pred_output "./image_predicted_car_segmented/" --flag_train 'False' --epochs 800 --checkpoint_dir './checkpoints_car_segmented/ckpt-31' --restore 'True' --evolution_images 10 --batch_size_train 10 --batch_size_test 1 --img_width 512 --img_height 512 --img_aug 40 --output_channels 3 --_lambda 100 
conda deactivate

conda activate tf_1_14_p_2_7
python2.7 enhancenet.py --path_input './image_predicted_car_segmented' --path_output './enhance_image_car_segmented'
conda deactivate



source ~/anaconda2/etc/profile.d/conda.sh
conda activate tf_2_p_3
python pix2pix_1.py --inpath "./train_input" --outpath './train_desired_output' --path_output "./output_model/" --predpath './image_to_pred_car' --pred_output "./image_predicted_car_not_segmented/" --flag_train 'False' --epochs 800 --checkpoint_dir './checkpoints_car_not_segmented/ckpt-9' --restore 'True' --evolution_images 10 --batch_size_train 10 --batch_size_test 1 --img_width 512 --img_height 512 --img_aug 40 --output_channels 3 --_lambda 100 

conda deactivate

conda activate tf_1_14_p_2_7
python2.7 enhancenet.py --path_input './image_predicted_car_not_segmented' --path_output './enhance_image_car_not_segmented'
conda deactivate
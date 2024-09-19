nvidia-smi
label=${0/launch_/}
label=${label/.sh/}
accelerate launch --mixed_precision bf16 train_controlnet.py --output_dir=output/${BASH_SOURCE/launch_/}/$(date -Ins) --validation_image {everett_kitchen{5,9},everett_lobby{1,16}}_dir_{23,14,18,10}.jpg --train_batch_size 8 --images_dir multilum_images/768x512 --inject_lighting_direction --concat_depth_maps --dropout_rgb 0.1 --dir_sh 4 --max_train_steps 200000 "${@}"
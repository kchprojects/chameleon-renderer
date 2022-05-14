#!/bin/sh
cd ../build
for dataset in pcb_bot_small coin_bot_small fi_rock_bot_small mag_box_bot_small ;  do
    ./validate_barytex ~/Diplomka/dataset_v2/$dataset/ ../results_x_switch/$dataset/
done./
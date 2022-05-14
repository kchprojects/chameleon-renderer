#!/bin/sh
cd ../build
for dataset in pcb_bot_small coin_bot_small fi_rock_bot_small mag_box_bot_small ;  do
    for forest in test_gs.lamb_forest test_col.lamb_forest test_col.bp_forest test_col.ct_forest;  do
        ./validate_show ~/Diplomka/dataset_v2/$dataset/ ../results/$dataset/$forest ../results/$dataset/render_views/$forest/
    done
done
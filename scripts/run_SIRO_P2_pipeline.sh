#!/usr/bin/env sh

# Snow-Informed Reservoir Operations (SIRO)
# USACE-ERDC-CRREL

SITE="kings"
DATA_DIR="/Users/rdcrlrka/Research/SIRO/SIRO_P2/study_sites/${SITE}"

python SIRO_P2_pipeline.py \
--model_dir $DATA_DIR/model_outputs \
--aoi_file ${DATA_DIR}/AOIs/${SITE}_watershed_outline_buffer10km/${SITE}_watershed_outline_buffer10km.shp \
--lidar_aoi_file ${DATA_DIR}/AOIs/${SITE}_watershed_outline/${SITE}_watershed_outline.shp \
--start_date "2022-10-01" \
--end_date "2025-06-30" \
--steps 2 4 \
--clip_fsca_to_aoi \
--remove_unclipped_fsca \
--out_dir $DATA_DIR
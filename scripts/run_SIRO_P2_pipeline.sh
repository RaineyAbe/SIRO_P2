#!/usr/bin/env sh

# Snow-Informed Reservoir Operations (SIRO)
# USACE-ERDC-CRREL

SITE="MCS"
DATA_DIR="/Users/rdcrlrka/Research/SIRO/SIRO_P2/study_sites/${SITE}"

python SIRO_P2_pipeline.py \
--model_dir $DATA_DIR/model_outputs \
--aoi_file ${DATA_DIR}/AOIs/${SITE}_watershed_outline_buffer10km/${SITE}_watershed_outline_buffer10km.shp \
--lidar_aoi_file ${DATA_DIR}/AOIs/${SITE}_watershed_outline/${SITE}_watershed_outline.shp \
--dem_file ${DATA_DIR}/static_inputs/${SITE}_3DEP_DEM_10m_buffer10km.tif \
--snotel_buffer_km 0 \
--start_date "2022-10-01" \
--end_date "2025-06-30" \
--steps 4 \
--clip_fsca_to_aoi \
--remove_unclipped_fsca \
--out_dir $DATA_DIR
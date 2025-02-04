#!/bin/bash

# INFO: carefully follow the instructions provided in "TODO: ..." sections.

# ****************************************************************************************
# **** CONFIGURE THE ENVIRONMENT *********************************************************
# ****************************************************************************************

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step 1. Configure the project directories
export DIR_PROJECT_ROOT= # TODO: set to project root directory
# /home/egor/Workspace

export DIR_DATA_ROOT=${DIR_PROJECT_ROOT}/data
export DIR_RESULTS_ROOT=${DIR_PROJECT_ROOT}/results
export DIR_CODE_ROOT=${DIR_PROJECT_ROOT}/koafusion
export DIR_CODE_ENTRY=${DIR_CODE_ROOT}/entry

# Step 2. Build Apptainer container
cd ${DIR_PROJECT_ROOT}
export DIR_TMP_ROOT=${DIR_PROJECT_ROOT}/tmp
mkdir ${DIR_TMP_ROOT}
APPTAINER_TMPDIR=${DIR_TMP_ROOT} apptainer build --nv apptainer.sif koagusion/apptainer.def

# Step 3. Try to execute a simple Python programme to ensure that Apptainer works
APPTAINER_FLAGS="--nv -B"
APPTAINER_FLAGS="${APPTAINER_FLAGS} ${DIR_RESULTS_ROOT}:/opt/results"
APPTAINER_FLAGS="${APPTAINER_FLAGS} ${DIR_DATA_ROOT}:/opt/data"

apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  python -c "import torch; print(torch.cuda.is_available())"
# ----------------------------------------------------------------------------------------

# ****************************************************************************************
# **** SELECT SAMPLES, DERIVE TARGETS, EXTRACT FROM THE OAI ******************************
# ****************************************************************************************

# Execute `koafusion/run/Targets_meta_and_scans_from_OAI.ipynb`
# TODO: specify the data paths and address other TODOs in the notebook step by step

# ****************************************************************************************
# **** PREPARE THE DATASET ***************************************************************
# ****************************************************************************************

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.prepare_data_xr_oulu \
   dir_root_mipt_xr=${DIR_DATA_ROOT}/OAI_XR_ROIs \
   dir_root_output=${DIR_DATA_ROOT}/OAI_XR_PA_prep \
   num_threads=24 \
)

apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.prepare_data_mri_oai \
  dir_root_oai_mri=${DIR_DATA_ROOT}/OAI_SAG_3D_DESS_raw \
  dir_root_output=${DIR_DATA_ROOT}/OAI_SAG_3D_DESS_prep \
  path_csv_extract=${DIR_DATA_ROOT}/meta_extract__sag_3d_dess.csv \
  num_threads=24 \
)

apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.prepare_data_mri_oai \
  dir_root_oai_mri=${DIR_DATA_ROOT}/OAI_COR_IW_TSE_raw \
  dir_root_output=${DIR_DATA_ROOT}/OAI_COR_IW_TSE_prep \
  path_csv_extract=${DIR_DATA_ROOT}/koafusion/meta_extract__cor_iw_tse.csv \
  num_threads=24 \
)

apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.prepare_data_mri_oai \
  dir_root_oai_mri=${DIR_DATA_ROOT}/OAI_SAG_T2_MAP_raw \
  dir_root_output=${DIR_DATA_ROOT}/OAI_SAG_T2_MAP_prep \
  path_csv_extract=${DIR_DATA_ROOT}/meta_extract__sag_t2_map.csv \
  num_threads=6 \
)
# ----------------------------------------------------------------------------------------

# ****************************************************************************************
# **** TRAIN THE MODELS ******************************************************************
# ****************************************************************************************

L_TARGETS=(prog_kl_12 prog_kl_24 prog_kl_36 prog_kl_48 tiulpin2019_prog_bin)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL=xr1_cnn
NUM_EPOCHS=60 ; BATCH_SIZE=64

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET} && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.arch=resnext50_32x4d model.fe.pretrained=true\
  model.input_size=[[700,700]] model.downscale=[[0.5,0.5]]\
  data.sets.n0.modals=[xr_pa] data.target="${TARGET}"\
  training.optim.lr_init=1e-3\
  +training.sched={name:CustomWarmupMultiStepLR,params:{epochs_warmup:5,mstep_milestones:[20,40]}}\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\
))
done
# ----------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL=mr1_cnn_trf
NUM_EPOCHS=60 ; BATCH_SIZE=32

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET}, sag_3d_dess && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.pretrained=true\
  model.agg.num_slices=64\
  model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]]\
  data.sets.n0.modals=[sag_3d_dess] data.target="${TARGET}"\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\
))
done

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET}, cor_iw_tse && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.pretrained=true\
  model.agg.num_slices=32\
  model.input_size=[[320,320,32]] model.downscale=[[0.5,0.5,1.0]]\
  data.sets.n0.modals=[cor_iw_tse] data.target="${TARGET}"\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\
))
done

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET}, sag_t2_map && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.pretrained=true\
  model.agg.num_slices=25\
  model.input_size=[[320,320,25]] model.downscale=[[0.5,0.5,1.0]]\
  data.sets.n0.modals=[sag_t2_map] data.target="${TARGET}"\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\
))
done
# ----------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL=mr2_cnn_trf
NUM_EPOCHS=60 ; BATCH_SIZE=16

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET}, sag_3d_dess cor_iw_tse && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.pretrained=true\
  model.input_size=[[320,320,128],[320,320,32]]\
  model.downscale=[[0.5,0.5,0.5],[0.5,0.5,1.0]]\
  model.agg.num_slices=[64,32]\
  data.sets.n0.modals=[sag_3d_dess,cor_iw_tse]\
  data.target="${TARGET}"\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\
)) done

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET}, sag_3d_dess sag_t2_map && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.pretrained=true\
  model.input_size=[[320,320,128],[320,320,25]]\
  model.downscale=[[0.5,0.5,0.5],[0.5,0.5,1.0]]\
  model.agg.num_slices=[64,25]\
  data.sets.n0.modals=[sag_3d_dess,sag_t2_map]\
  data.target="${TARGET}"\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\
)) done

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET}, cor_iw_tse sag_t2_map && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.pretrained=true\
  model.input_size=[[320,320,32],[320,320,25]]\
  model.downscale=[[0.5,0.5,1.0],[0.5,0.5,1.0]]\
  model.agg.num_slices=[32,25]\
  data.sets.n0.modals=[cor_iw_tse,sag_t2_map]\
  data.target="${TARGET}"\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\
)) done
# ----------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL=xr1mr1_cnn_trf
NUM_EPOCHS=60 ; BATCH_SIZE=32

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET}, xr_pa sag_3d_dess && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true\
  model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true\
  model.input_size=[[700,700],[320,320,128]]\
  model.downscale=[[0.5,0.5],[0.5,0.5,0.5]]\
  model.agg.num_slices=[1,64]\
  data.sets.n0.modals=[xr_pa,sag_3d_dess]\
  data.target="${TARGET}"\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\

))
done

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET}, xr_pa cor_iw_tse && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true\
  model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true\
  model.input_size=[[700,700],[320,320,32]]\
  model.downscale=[[0.5,0.5],[0.5,0.5,1.0]]\
  model.agg.num_slices=[1,32]\
  data.sets.n0.modals=[xr_pa,cor_iw_tse]\
  data.target="${TARGET}"\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\
))
done

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET}, xr_pa sag_t2_map && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true\
  model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true\
  model.input_size=[[700,700],[320,320,25]]\
  model.downscale=[[0.5,0.5],[0.5,0.5,1.0]]\
  model.agg.num_slices=[1,25]\
  data.sets.n0.modals=[xr_pa,sag_t2_map]\
  data.target="${TARGET}"\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\
))
done
# ----------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL=xr1mr2_cnn_trf
NUM_EPOCHS=60 ; BATCH_SIZE=16

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET}, xr_pa sag_3d_dess cor_iw_tse && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true\
  model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true\
  model.input_size=[[700,700],[320,320,128],[320,320,32]]\
  model.downscale=[[0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,1.0]]\
  model.agg.num_slices=[1,64,32]\
  data.sets.n0.modals=[xr_pa,sag_3d_dess,cor_iw_tse]\
  data.target="${TARGET}"\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\
))
done

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET}, xr_pa sag_3d_dess sag_t2_map && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true\
  model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true\
  model.input_size=[[700,700],[320,320,128],[320,320,25]]\
  model.downscale=[[0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,1.0]]\
  model.agg.num_slices=[1,64,25]\
  data.sets.n0.modals=[xr_pa,sag_3d_dess,sag_t2_map]\
  data.target="${TARGET}"\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\
))
done

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET}, xr_pa cor_iw_tse sag_t2_map && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
  path_project_root=/opt path_data_root=/opt/data\
  model="${MODEL}"\
  model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true\
  model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true\
  model.input_size=[[700,700],[320,320,32],[320,320,25]]\
  model.downscale=[[0.5,0.5],[0.5,0.5,1.0],[0.5,0.5,1.0]]\
  model.agg.num_slices=[1,32,25]\
  data.sets.n0.modals=[xr_pa,cor_iw_tse,sag_t2_map]\
  data.target="${TARGET}"\
  training.epochs.num="${NUM_EPOCHS}"\
  training.batch_size="${BATCH_SIZE}"\
  validation.batch_size="${BATCH_SIZE}"\
))
done
# ----------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL=xr1mr2c1_cnn_trf
NUM_EPOCHS=60 ; BATCH_SIZE=16

for TARGET in "${L_TARGETS[@]}"; do
(echo ${MODEL}, ${TARGET} && \
 apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
(python -m koafusion.train_prog_fus \
 path_project_root=/opt path_data_root=/opt/data \
 model="${MODEL}" \
 model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true \
 model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true \
 model.fe.xr.dropout=0.1 model.fe.mr.dropout=0.1 model.fe.clin.dropout=0.1 \
 model.fe.clin.dim_in=9 model.fe.clin.dim_out=2048 \
 model.input_size=[[700,700],[320,320,128],[320,320,25],[16]] \
 model.downscale=[[0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,1.0],[1.0]] \
 model.agg.num_slices=[1,64,25,1] \
 data.sets.n0.modals=[xr_pa,sag_3d_dess,sag_t2_map,clin] \
 data.target="${TARGET}" \
 training.epochs.num="${NUM_EPOCHS}" \
 training.batch_size="${BATCH_SIZE}" \
 validation.batch_size="${BATCH_SIZE}" \
))
done
# ----------------------------------------------------------------------------------------

# ****************************************************************************************
# **** EVALUATE AND EXPLAIN THE IMAGING MODELS *******************************************
# ****************************************************************************************

# TODO: all examples below are for evaluation regime (i.e. making model predictions);
# TODO: for explanation regime (i.e. to derive modal ablation data), uncomment
# TODO: ARGS_EVAL_REGIME below and execute again.
#ARGS_EVAL_REGIME=(
#"model.output_type=main"
#"testing.regime=explain"
#"testing.explain_fn=modal_abl"
#"testing.use_cached=true"
#)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL=xr1_cnn
BATCH_SIZE=32

# TODO: uncomment one at a time
#TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model="${MODEL}" \
  model.fe.arch=resnext50_32x4d model.fe.pretrained=true \
  model.restore_weights=true \
  model.input_size=[[700,700]] model.downscale=[[0.5,0.5]] \
  data.sets.n0.modals=[xr_pa] data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  testing.profile=${PROFILE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))
# ----------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL=mr1_cnn_trf
BATCH_SIZE=32

# TODO: uncomment one at a time
#TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model="${MODEL}" \
  model.fe.pretrained=true \
  model.restore_weights=true \
  model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]] \
  data.sets.n0.modals=[sag_3d_dess] data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))

# TODO: uncomment one at a time
TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model="${MODEL}" \
  model.fe.pretrained=true \
  model.restore_weights=true \
  model.input_size=[[320,320,32]] model.downscale=[[0.5,0.5,1.0]] \
  data.sets.n0.modals=[cor_iw_tse] data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))

# TODO: uncomment one at a time
TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model="${MODEL}" \
  model.fe.pretrained=true \
  model.restore_weights=true \
  model.input_size=[[320,320,25]] model.downscale=[[0.5,0.5,1.0]] \
  data.sets.n0.modals=[sag_t2_map] data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))
# ----------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL=mr2_cnn_trf
BATCH_SIZE=8

# TODO: uncomment one at a time
TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model=${MODEL} \
  model.fe.pretrained=true \
  model.restore_weights=true \
  model.input_size=[[320,320,128],[320,320,32]] \
  model.downscale=[[0.5,0.5,0.5],[0.5,0.5,1.0]] \
  model.agg.num_slices=[64,32] \
  data.sets.n0.modals=[sag_3d_dess,cor_iw_tse] \
  data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))

# TODO: uncomment one at a time
TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model=${MODEL} \
  model.fe.pretrained=true \
  model.restore_weights=true \
  model.input_size=[[320,320,128],[320,320,25]] \
  model.downscale=[[0.5,0.5,0.5],[0.5,0.5,1.0]] \
  model.agg.num_slices=[64,25] \
  data.sets.n0.modals=[sag_3d_dess,sag_t2_map] \
  data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))

# TODO: uncomment one at a time
TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model=${MODEL} \
  model.fe.pretrained=true \
  model.restore_weights=true \
  model.input_size=[[320,320,32],[320,320,25]] \
  model.downscale=[[0.5,0.5,1.0],[0.5,0.5,1.0]] \
  model.agg.num_slices=[32,25] \
  data.sets.n0.modals=[cor_iw_tse,sag_t2_map] \
  data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))
# ----------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL=xr1mr2_cnn_trf
BATCH_SIZE=8

# TODO: uncomment one at a time
#TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model.restore_weights=true \
  model="${MODEL}"\
  model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true \
  model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true \
  model.input_size=[[700,700],[320,320,128],[320,320,32]] \
  model.downscale=[[0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,1.0]] \
  model.agg.num_slices=[1,64,32] \
  data.sets.n0.modals=[xr_pa,sag_3d_dess,cor_iw_tse] \
  data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))

# TODO: uncomment one at a time
#TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model.restore_weights=true \
  model="${MODEL}" \
  model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true \
  model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true \
  model.input_size=[[700,700],[320,320,128],[320,320,25]] \
  model.downscale=[[0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,1.0]] \
  model.agg.num_slices=[1,64,25] \
  data.sets.n0.modals=[xr_pa,sag_3d_dess,sag_t2_map] \
  data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))

# TODO: uncomment one at a time
#TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model.restore_weights=true \
  model="${MODEL}" \
  model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true \
  model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true \
  model.input_size=[[700,700],[320,320,32],[320,320,25]] \
  model.downscale=[[0.5,0.5],[0.5,0.5,1.0],[0.5,0.5,1.0]] \
  model.agg.num_slices=[1,32,25] \
  data.sets.n0.modals=[xr_pa,cor_iw_tse,sag_t2_map] \
  data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))
# ----------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL=xr1mr1_cnn_trf
BATCH_SIZE=8

# TODO: uncomment one at a time
#TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model.restore_weights=true \
  model="${MODEL}"\
  model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true \
  model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true \
  model.input_size=[[700,700],[320,320,128]] \
  model.downscale=[[0.5,0.5],[0.5,0.5,0.5]] \
  model.agg.num_slices=[1,64] \
  data.sets.n0.modals=[xr_pa,sag_3d_dess] \
  data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))

# TODO: uncomment one at a time
#TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model.restore_weights=true \
  model="${MODEL}" \
  model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true \
  model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true \
  model.input_size=[[700,700],[320,320,32]] \
  model.downscale=[[0.5,0.5],[0.5,0.5,1.0]] \
  model.agg.num_slices=[1,32] \
  data.sets.n0.modals=[xr_pa,cor_iw_tse] \
  data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))

# TODO: uncomment one at a time
TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
  path_project_root=/opt path_data_root=/opt/data \
  path_experiment_root=/opt/results/${EXPERIM} \
  model.restore_weights=true \
  model="${MODEL}" \
  model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true \
  model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true \
  model.input_size=[[700,700],[320,320,25]] \
  model.downscale=[[0.5,0.5],[0.5,0.5,1.0]] \
  model.agg.num_slices=[1,25] \
  data.sets.n0.modals=[xr_pa,sag_t2_map] \
  data.target="${TARGET}" \
  testing.batch_size=${BATCH_SIZE} \
  data.ignore_cache=true \
  "${ARGS_EVAL_REGIME[@]}" \
))
# ----------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL=xr1mr2c1_cnn_trf
BATCH_SIZE=8

# TODO: uncomment one at a time
#TARGET=prog_kl_12 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_24 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_36 ; EXPERIM= # TODO: specify experiment_id
#TARGET=prog_kl_48 ; EXPERIM= # TODO: specify experiment_id
#TARGET=tiulpin2019_prog_bin ; EXPERIM= # TODO: specify experiment_id

(echo ${TARGET} ${EXPERIM} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.eval_prog_fus \
    path_project_root=/opt path_data_root=/opt/data \
    path_experiment_root=/opt/results/${EXPERIM} \
    model.restore_weights=true \
    model="${MODEL}" \
    model.fe.xr.arch=resnext50_32x4d model.fe.xr.pretrained=true \
    model.fe.mr.arch=resnet50 model.fe.mr.pretrained=true \
    model.fe.clin.dim_in=9 model.fe.clin.dim_out=2048 \
    model.input_size=[[700,700],[320,320,128],[320,320,25],[16]] \
    model.downscale=[[0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,1.0],[1.0]] \
    model.agg.num_slices=[1,64,25,1] \
    data.sets.n0.modals=[xr_pa,sag_3d_dess,sag_t2_map,clin] \
    data.target="${TARGET}" \
    testing.batch_size=${BATCH_SIZE} \
    data.ignore_cache=true \
    "${ARGS_EVAL_REGIME[@]}" \
))
# ----------------------------------------------------------------------------------------

# ****************************************************************************************
# **** TRAIN AND EVALUATE THE CLINICAL MODELS ********************************************
# ****************************************************************************************

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TODO: uncomment one at a time
#L_VARS="[age,sex,bmi]"
#L_VARS="[age,sex,bmi,kl]"
#L_VARS="[age,sex,bmi,surj,inj,womac]"
#L_VARS="[age,sex,bmi,kl,surj,inj,womac]"

for TARGET in "${L_TARGETS[@]}"
do
(echo "train_prog_clin" &&
LABEL_TIME=`date +%y%m%d_%H%M` &&
LABEL_RAND=`openssl rand -hex 2` &&
EXPERIM=${LABEL_TIME}__${LABEL_RAND} &&
  apptainer exec ${APPTAINER_FLAGS} apptainer.sif \
  (python -m koafusion.train_prog_clin \
  path_project_root=/opt path_data_root=/opt/data \
  experiment_id=${EXPERIM} \
  model.vars=${L_VARS} \
  data.target=${TARGET} \
  data.ignore_cache=true \
  validation.criterion=balanced_accuracy \
))
done
# ----------------------------------------------------------------------------------------

# ****************************************************************************************
# **** ANALYZE THE RESULTS AND COMPARE THE MODELS ****************************************
# ****************************************************************************************

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO: specify the experiment_id's and address other TODOs in the notebook
# Execute `koafusion/run/Analysis_Visualization.ipynb` step by step

# ----------------------------------------------------------------------------------------

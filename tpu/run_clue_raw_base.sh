CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
CURRENT_TIME=$(date "+%Y%m%d-%H%M%S")

run_task() {
  TASK_NAME=$1
  export PREV_TRAINED_MODEL_DIR=gs://models_zxw/prev_trained_models/nlp/official_albert/$2
  export DATA_DIR=gs://data_zxw/nlp/CLUE/${TASK_NAME}_public
  export OUTPUT_DIR=gs://models_zxw/fine_tuning_models/nlp/official_albert/$2/tpu/raw/$TASK_NAME/$CURRENT_TIME
  COMMON_ARGS="
    --data_dir=$DATA_DIR \
    --do_lower_case \
    --optimizer=adamw \
    --vocab_file=$PREV_TRAINED_MODEL_DIR/vocab_chinese.txt \
    --albert_config_file=$PREV_TRAINED_MODEL_DIR/albert_config.json \
    --task_name=$TASK_NAME \
    --max_seq_length=$3 \
    --warmup_step=$4 \
    --learning_rate=$5 \
    --train_step=$6 \
    --save_checkpoints_steps=$7 \
    --train_batch_size=$8 \
    --output_dir=$OUTPUT_DIR \
    --num_tpu_cores=8 --use_tpu=True --tpu_name=grpc://$9:8470
  "
  python3 $CURRENT_DIR/../raw_run_classifier.py \
    $COMMON_ARGS \
    --do_train \
    --nodo_eval \
    --nodo_predict \
    --init_checkpoint=$PREV_TRAINED_MODEL_DIR/model.ckpt-best 

  python3 $CURRENT_DIR/../raw_run_classifier.py \
    $COMMON_ARGS \
    --nodo_train \
    --do_eval \
    --do_predict 
}
# command # task # model # max_seq_length # warmup_step # learning_rate # train_step # tpu
#run_task afqmc albert_base 128 300 1e-5 7000 100 16 10.230.11.170
#run_task tnews albert_base 128 300 1e-5 40000 100 16 10.230.11.170
run_task iflytek albert_base 128 300 1e-5 20000 100 16 10.237.140.202
#run_task wsc albert_base 128 10 1e-5 100 10 16 10.237.140.202
#run_task csl albert_base 128 300 1e-5 4000 100 16 10.237.140.202
#run_task cmnli albert_base 128 300 1e-5 100000 500 16 10.237.140.202

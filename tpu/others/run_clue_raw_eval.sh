CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
CURRENT_TIME=20200121-150605
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
    --max_seq_length=$3 \
    --train_batch_size=$4 \
    --learning_rate=$5 \
    --num_train_epochs=$6 \
    --save_checkpoints_steps=$7 \
    --output_dir=$OUTPUT_DIR \
    --num_tpu_cores=8 --use_tpu=True --tpu_name=grpc://$8:8470
  "

  python3 $CURRENT_DIR/../run_classifier.py \
    $COMMON_ARGS \
    --task_name=$TASK_NAME \
    --nodo_train \
    --do_eval \
    --do_predict 
}

#run_task afqmc albert_xlarge 128 32 1e-5 3 500 10.237.140.202
#run_task tnews albert_xlarge 128 32 1e-5 3 500 10.237.140.202
#run_task iflytek albert_xlarge 128 32 1e-5 3 500 10.237.140.202
run_task cmnli albert_xlarge 128 32 1e-5 3 500 10.237.140.202
#run_task wsc albert_xlarge 128 32 1e-5 3 500 10.237.140.202
#run_task csl albert_xlarge 128 32 1e-5 3 500 10.237.140.202

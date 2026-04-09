
set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
MAX_TOKENS_PER_CALL=$3
OUTPUT_DIR=$4

SPLIT="test"
NUM_TEST_SAMPLE=-1
DATA_NAMES="math500"

# Skip if already evaluated
if [ -d "${OUTPUT_DIR}/${DATA_NAMES}" ] && \
   [ -n "$(find ${OUTPUT_DIR}/${DATA_NAMES} -name '*metrics.json' -print -quit)" ]; then
    echo "============ math500 already evaluated — showing results ============="
    METRICS_FILE=$(find ${OUTPUT_DIR}/${DATA_NAMES} -name '*metrics.json' -print -quit)
    cat ${METRICS_FILE}
else
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name ${DATA_NAMES} \
        --output_dir ${OUTPUT_DIR} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0 \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --max_tokens_per_call ${MAX_TOKENS_PER_CALL} \
        --overwrite
fi

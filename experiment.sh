#!/bin/bash

# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### Required parameters (modify before calling the script!) ###

NUM_EPOCHS=20
EXPORT_CHECKPOINT=0
BATCH_SIZE=4
PHRASE_VOCAB_SIZE=500
MAX_INPUT_EXAMPLES=1000000
SAVE_CHECKPOINT_STEPS=260
MAX_SEQ_LENGTH=512

# Download the WikiSplit data from:
# https://github.com/google-research-datasets/wiki-split
WIKISPLIT_DIR=data
# Preprocessed data and models will be stored here.
OUTPUT_DIR=out
# Download the pretrained BERT model:
# https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
BERT_BASE_DIR=~/cased_L-12_H-768_A-12

### Optional parameters ###

# If you train multiple models on the same data, change this label.
EXPERIMENT=sec_letter_experiment


###########################

OPTIMIZE=0
PREPROCESS=0
TRAIN=0
VALIDATE=0
EXPORT=0
PREDICT=0
SCORE=0

if [ $# -eq 0 ]
then
	OPTIMIZE=1
	PREPROCESS=1
	TRAIN=1
	EXPORT=1
	PREDICT=1
	SCORE=1
elif [ "$1" = "optimize" ]
then
    OPTIMIZE=1
elif [ "$1" = "preprocess" ]
then
    PREPROCESS=1
elif [ "$1" = "train" ]
then
    TRAIN=1
elif [ "$1" = "validate" ]
then
    VALIDATE=1
elif [ "$1" = "export" ]
then
    EXPORT=1
elif [ "$1" = "predict" ]
then
    PREDICT=1
elif [ "$1" = "score" ]
then
    SCORE=1
fi

### 1. Phrase Vocabulary Optimization

if [ $OPTIMIZE -eq 1 ]
then
	python phrase_vocabulary_optimization.py \
		--input_file=${WIKISPLIT_DIR}/train.tsv \
		--input_format=wikisplit \
		--vocabulary_size=${PHRASE_VOCAB_SIZE} \
		--max_input_examples=${MAX_INPUT_EXAMPLES} \
		--output_file=${OUTPUT_DIR}/label_map.txt \
		--noenable_swap_tag
fi

### 2. Converting Target Texts to Tags

if [ $PREPROCESS -eq 1 ]
then
	python preprocess_main.py \
		--input_file=${WIKISPLIT_DIR}/tune.tsv \
		--input_format=wikisplit \
		--output_tfrecord=${OUTPUT_DIR}/tune.tf_record \
		--label_map_file=${OUTPUT_DIR}/label_map.txt \
		--vocab_file=${BERT_BASE_DIR}/vocab.txt \
		--output_arbitrary_targets_for_infeasible_examples=true \
		--max_seq_length=${MAX_SEQ_LENGTH} \
		--noenable_swap_tag

	python preprocess_main.py \
		--input_file=${WIKISPLIT_DIR}/train.tsv \
		--input_format=wikisplit \
		--output_tfrecord=${OUTPUT_DIR}/train.tf_record \
		--label_map_file=${OUTPUT_DIR}/label_map.txt \
		--vocab_file=${BERT_BASE_DIR}/vocab.txt \
		--output_arbitrary_targets_for_infeasible_examples=false \
		--max_seq_length=${MAX_SEQ_LENGTH} \
		--noenable_swap_tag
fi

### 3. Model Training

CONFIG_FILE=./configs/lasertagger_config.json

if [ $TRAIN -eq 1 ] || [ $VALIDATE -eq 1 ]
then
	NUM_TRAIN_EXAMPLES=$(cat "${OUTPUT_DIR}/train.tf_record.num_examples.txt")
	NUM_EVAL_EXAMPLES=$(cat "${OUTPUT_DIR}/tune.tf_record.num_examples.txt")

	DO_TRAIN=false
	if [ $TRAIN -eq 1 ]
	then
	  DO_TRAIN=true
	fi

	python run_lasertagger.py \
		--training_file=${OUTPUT_DIR}/train.tf_record \
		--eval_file=${OUTPUT_DIR}/tune.tf_record \
		--label_map_file=${OUTPUT_DIR}/label_map.txt \
		--model_config_file=${CONFIG_FILE} \
		--output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
		--init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
		--do_train=${DO_TRAIN} \
		--do_eval=true \
		--train_batch_size=${BATCH_SIZE} \
		--save_checkpoints_steps=${SAVE_CHECKPOINT_STEPS} \
		--num_train_epochs=${NUM_EPOCHS} \
		--num_train_examples=${NUM_TRAIN_EXAMPLES} \
		--num_eval_examples=${NUM_EVAL_EXAMPLES} \
		--max_seq_length=${MAX_SEQ_LENGTH} \
		--noenable_swap_tag
fi

### 4. Prediction

# Export the model.
if [ $EXPORT -eq 1 ]
then
	python run_lasertagger.py \
		--label_map_file=${OUTPUT_DIR}/label_map.txt \
		--model_config_file=${CONFIG_FILE} \
		--output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
		--do_export=true \
		--export_path=${OUTPUT_DIR}/models/${EXPERIMENT}/export \
    --init_checkpoint=${OUTPUT_DIR}/models/${EXPERIMENT}/model.ckpt-${EXPORT_CHECKPOINT}
fi

PREDICTION_FILE=${OUTPUT_DIR}/models/${EXPERIMENT}/pred.tsv

if [ $PREDICT -eq 1 ]
then
	# Get the most recently exported model directory.
	TIMESTAMP=$(ls "${OUTPUT_DIR}/models/${EXPERIMENT}/export/" | grep -v "temp-" | sort -r | head -1)
	SAVED_MODEL_DIR=${OUTPUT_DIR}/models/${EXPERIMENT}/export/${TIMESTAMP}

	python predict_main.py \
		--input_file=${WIKISPLIT_DIR}/validation.tsv \
		--input_format=wikisplit \
		--output_file=${PREDICTION_FILE} \
		--label_map_file=${OUTPUT_DIR}/label_map.txt \
		--vocab_file=${BERT_BASE_DIR}/vocab.txt \
		--saved_model=${SAVED_MODEL_DIR} \
		--max_seq_length=${MAX_SEQ_LENGTH} \
		--noenable_swap_tag
fi

### 5. Evaluation

if [ $SCORE -eq 1 ]
then
	python score_main.py --prediction_file=${PREDICTION_FILE}
fi

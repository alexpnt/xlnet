import itertools
import os
import uuid
from functools import partial

import numpy as np
import sentencepiece as spm
import tensorflow as tf

import xlnet
from classifier_utils import convert_single_example
from model_utils import init_from_checkpoint, get_assignment_map_from_checkpoint, configure_tpu
from prepro_utils import preprocess_text, encode_ids
from run_classifier import InputExample


# enable eager execution
# tf.enable_eager_execution()
# tf.random.set_random_seed(0)

# build a tokenizer based on the sentence piece model
def tokenize_fn_builder(spiece_model_path):
    sentence_piece_model = spm.SentencePieceProcessor()
    sentence_piece_model.Load(spiece_model_path)

    def tokenize_fn(text, sentence_piece_model):
        text = preprocess_text(text, lower=False)
        return encode_ids(sentence_piece_model, text)

    return partial(tokenize_fn, sentence_piece_model=sentence_piece_model)


def configure_tpu(model_dir, num_core_per_host=1, iterations=1000, num_hosts=1, max_save=0, save_steps=None):
    master = None

    session_config = tf.ConfigProto(allow_soft_placement=True)
    # Uncomment the following line if you hope to monitor GPU RAM growth
    # session_config.gpu_options.allow_growth = True

    if num_core_per_host == 1:
        strategy = None
        tf.logging.info('Single device mode.')
    else:
        strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=num_core_per_host)
        tf.logging.info('Use MirroredStrategy with %d devices.', strategy.num_replicas_in_sync)

    per_host_input = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=master,
        model_dir=model_dir,
        session_config=session_config,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations,
            num_shards=num_hosts * num_core_per_host,
            per_host_input_for_training=per_host_input),
        keep_checkpoint_max=max_save,
        save_checkpoints_secs=None,
        save_checkpoints_steps=save_steps,
        train_distribute=strategy
    )
    return run_config


def init_from_checkpoint(init_checkpoint, global_vars=False):
    tvars = tf.global_variables() if global_vars else tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint is not None:
        if init_checkpoint.endswith("latest"):
            ckpt_dir = os.path.dirname(init_checkpoint)
            init_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        else:
            init_checkpoint = init_checkpoint

        tf.logging.info("Initialize from the ckpt {}".format(init_checkpoint))

        (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # Log customized initialization
        tf.logging.info("**** Global Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)


def input_fn_builder(input_examples, tokenize_fn, max_seq_lenght, batch_size):
    input_features_ds = {
        "input_ids": [],
        "segment_ids": [],
        "input_mask": [],
    }
    for inp_example in input_examples:
        input_features = convert_single_example(uuid.uuid4().int, inp_example, label_list=None,
                                                max_seq_length=max_seq_lenght, tokenize_fn=tokenize_fn)
        input_features_ds["input_ids"] += [input_features.input_ids]
        input_features_ds["segment_ids"] += [input_features.segment_ids]
        input_features_ds["input_mask"] += [input_features.input_mask]

    def input_fn():
        dataset = {'input_ids': tf.reshape(tf.constant(input_features_ds['input_ids'], tf.int32),
                                           [max_seq_lenght, batch_size]),
                   'seg_ids': tf.reshape(tf.constant(input_features_ds['segment_ids'], tf.int32),
                                         [max_seq_lenght, batch_size]),
                   'input_mask': tf.reshape(tf.constant(input_features_ds['input_mask'], tf.float32),
                                            [max_seq_lenght, batch_size])}

        return dataset

    return input_fn


def model_fn_builder(model_config_path, model_ckpt_path):
    def model_fn(features):
        mode = tf.estimator.ModeKeys.PREDICT

        # XLNetConfig contains hyperparameters that are specific to a model checkpoint
        xlnet_config = xlnet.XLNetConfig(json_path=model_config_path)
        xlnet_config.dropout = 0.0
        xlnet_config.dropatt = 0.0

        # RunConfig contains hyperparameters that could be different between pretraining and finetuning.
        xlnet_run_config = xlnet.RunConfig(is_training=False, use_tpu=False, use_bfloat16=False, dropout=0.0,
                                           dropatt=0.0, init="normal", init_range=0.1, init_std=0.02, mem_len=None,
                                           reuse_len=None, bi_data=False, clamp_len=-1, same_length=False)

        # Construct an XLNet model
        xlnet_model = xlnet.XLNetModel(
            xlnet_config=xlnet_config,
            run_config=xlnet_run_config,
            input_ids=features['input_ids'],
            seg_ids=features['seg_ids'],
            input_mask=features['input_mask']
        )

        #### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        #### load pretrained models
        init_from_checkpoint(model_ckpt_path)

        # Get a summary of the sequence using the last hidden state
        summary = xlnet_model.get_pooled_out(summary_type="last", use_summ_proj=True)

        # Get a sequence output
        # seq_out = xlnet_model.get_sequence_output()

        output = {
            'summary': summary,
            # 'seq_out': seq_out
        }

        output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=output)
        return output_spec

    return model_fn


def encode_sentences(sentences, tokenize_fn, max_seq_length, model_config_path, model_ckpt_path, model_finetune_dir):
    input_examples = [InputExample(guid=uuid.uuid4(), text_a=sent, text_b=None, label=None) for sent in sentences]
    batch_size = len(input_examples)

    run_config = configure_tpu(model_finetune_dir)
    model_fn = model_fn_builder(model_config_path, model_ckpt_path)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    input_fn = input_fn_builder(
        input_examples=input_examples,
        tokenize_fn=tokenize_fn,
        max_seq_lenght=max_seq_length,
        batch_size=batch_size
    )

    encoder = estimator.predict(input_fn=input_fn, yield_single_examples=False, checkpoint_path=None)
    results = list(itertools.islice(encoder, 1))[0]
    # print(results['summary'])
    return results['summary']


if __name__ == "__main__":
    model_base_path = '/xlnet_cased_L-12_H-768_A-12/'
    model_config_path = model_base_path + 'xlnet_config.json'
    model_ckpt_path = model_base_path + 'xlnet_model.ckpt'
    spiece_model_path = model_base_path + 'spiece.model'
    model_finetune_dir = model_base_path + 'finetuned/'

    max_seq_length = 512

    tokenize_fn = tokenize_fn_builder(spiece_model_path)
    encode_sentences(['this is a test', 'And this is another test', 'Yet another random tests'], tokenize_fn,
                     max_seq_length, model_config_path, model_ckpt_path, model_finetune_dir)

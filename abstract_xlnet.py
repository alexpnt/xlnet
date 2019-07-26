import uuid
from functools import partial

import sentencepiece as spm
import tensorflow as tf

import xlnet
from classifier_utils import convert_single_example
from prepro_utils import preprocess_text, encode_ids
from run_classifier import InputExample

# enable eager execution
# tf.enable_eager_execution()


def build_xlnet_config(model_config_path, spiece_model_path):
    # build a tokenizer based on the sentence piece model
    def tokenize_fn(text, sentence_piece_model):
        text = preprocess_text(text, lower=False)
        return encode_ids(sentence_piece_model, text)

    # XLNetConfig contains hyperparameters that are specific to a model checkpoint.
    xlnet_config = xlnet.XLNetConfig(json_path=model_config_path)
    xlnet_config.dropout = 0.0
    xlnet_config.dropatt = 0.0

    # RunConfig contains hyperparameters that could be different between pretraining and finetuning.
    xlnet_run_config = xlnet.RunConfig(is_training=False, use_tpu=False, use_bfloat16=False, dropout=0.0, dropatt=0.0,
                                       init="normal", init_range=0.1, init_std=0.02, mem_len=None,
                                       reuse_len=None, bi_data=False, clamp_len=-1, same_length=False)

    sentence_piece_model = spm.SentencePieceProcessor()
    sentence_piece_model.Load(spiece_model_path)

    return {'model_config': xlnet_config,
            'run_config': xlnet_run_config,
            'tokenizer': partial(tokenize_fn, sentence_piece_model=sentence_piece_model)
            }


def encode_sentences(sentences, xlnet_config, xlnet_run_config, tokenize_fn, max_seq_lenght):
    batch_size = len(sentences)

    input_examples = [InputExample(guid=uuid.uuid4(), text_a=sent, text_b=None, label=None) for sent in sentences]

    input_features_batch = {
        "input_ids": [],
        "segment_ids": [],
        "input_mask": [],
    }
    for inp_example in input_examples:
        input_features = convert_single_example(uuid.uuid4().int, inp_example, label_list=None,
                                                max_seq_length=max_seq_lenght, tokenize_fn=tokenize_fn)
        input_features_batch["input_ids"] += [input_features.input_ids]
        input_features_batch["segment_ids"] += [input_features.segment_ids]
        input_features_batch["input_mask"] += [input_features.input_mask]

    # Construct an XLNet model
    xlnet_model = xlnet.XLNetModel(
        xlnet_config=xlnet_config,
        run_config=xlnet_run_config,
        input_ids=tf.reshape(tf.constant(input_features_batch['input_ids'], tf.int32), [max_seq_lenght, batch_size]),
        seg_ids=tf.reshape(tf.constant(input_features_batch['segment_ids'], tf.int32), [max_seq_lenght, batch_size]),
        input_mask=tf.reshape(tf.constant(input_features_batch['input_mask'], tf.float32), [max_seq_lenght, batch_size])
    )

    # Get a summary of the sequence using the last hidden state
    pooled_output = xlnet_model.get_pooled_out(summary_type="last", use_summ_proj=False)
    print(pooled_output)
    print(pooled_output.shape)
    return pooled_output


if __name__ == "__main__":
    model_base_path = 'data/models/xlnet/xlnet_cased_L-12_H-768_A-12/'
    model_config_path = model_base_path + 'xlnet_config.json'
    model_ckpt_path = model_base_path + 'xlnet_model.ckpt'
    model_finetune_dir = model_base_path + 'finetuned/'
    spiece_model_path = model_base_path + 'spiece.model'
    max_seq_length = 512

    config = build_xlnet_config(model_config_path, spiece_model_path)
    encode_sentences(['this is a test', 'And this is another test'],
                     config['model_config'],
                     config['run_config'],
                     config['tokenizer'],
                     max_seq_length)

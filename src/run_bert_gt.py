# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import logging
import os
import random
#import tensorflow as tf
import tensorflow.compat.v1 as tf

import modeling_gt as modeling
import optimization
import tokenization_gt as tokenization

import graph_algorithm

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "test_file", "",
    "The input path of test set file.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "use_balanced_neg", False,
    "Whether to balance the numbers of negative and non-negative instances in train/dev?")

flags.DEFINE_bool(
    "no_neg_for_train_dev", False,
    "Not to use negative instances in train and dev")

flags.DEFINE_bool(
    "test_has_header", True,
    "Whether the test set has a header or not?")

flags.DEFINE_integer(
    "max_neg_scale", 2,
    "The times of negative instances over the other instances. It is used only if use_balanced_neg == True ")


flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_num_entity_indices", 50,
    "The maximum number of indices for each entity.")

flags.DEFINE_integer("max_num_neighbors", 10, "Maximum number of neighbors.")

flags.DEFINE_integer("gt_hidden_size", 768, "Hidden layer size of graph transformer.")

flags.DEFINE_integer("num_gt_hidden_layers", 12, "Hidden layer number of graph transformer.")

flags.DEFINE_integer("num_gt_attention_heads", 12, "Attention head number of graph transformer.")

flags.DEFINE_integer("gt_intermediate_size", 3072, "Intermediate size of graph transformer.")

flags.DEFINE_bool(
    "shortest_path_neighbors", True,
    "extend entities' neighbors by adding their neighbors which are in the shortest path to other entities")

flags.DEFINE_integer(
    "surrounding_words_distance", 1,
    "extend neighbors by adding their surrounding words inside the specified distance")

flags.DEFINE_bool(
    "use_entities_as_neighbors", True,
    "extend neighbors by adding all entities indices with the same entity type")

flags.DEFINE_integer("entity_num", 2, "Number of entities.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, 
               guid, 
               text_a, 
               in_neighbors=None, 
               text_b=None, 
               label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.in_neighbors = in_neighbors
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               input_graph,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.input_graph = input_graph
    self.label_id = label_id
    self.is_real_example = is_real_example


class InputGraph(object):

  def __init__(self,
               entity_indices,
               in_neighbors_indices,
               entity_indices_mask,
               in_neighbors_indices_mask):
    
    self.entity_indices = entity_indices
    self.in_neighbors_indices = in_neighbors_indices
    
    self.entity_indices_mask = entity_indices_mask
    self.in_neighbors_indices_mask = in_neighbors_indices_mask

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, has_header):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()
        
    def get_test_examples_by_file(self, in_test_file):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()
        
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_entity_type_dict(self):
        raise NotImplementedError()
    
    def get_entity_type_list(self):
        return sorted([entity_type for entity_type in self.get_entity_type_dict().keys()])

    def get_entity_indices_by_types(self, text_a):
        entity_type_dict = self.get_entity_type_dict()
        all_indices = {}
        i_wo_empty_string = -1
        for i, token in enumerate(text_a.split(' ')):
            if token != '':
                i_wo_empty_string += 1
            if token in entity_type_dict:
                if token not in all_indices:
                    all_indices[token] = []
                #all_indices[token].append(i)
                all_indices[token].append(i_wo_empty_string)
        return all_indices
    
    def get_entity_types_in_text(self, text_a):
        entity_type_dict = self.get_entity_type_dict()
        entity_types_in_text = set()
        for i, token in enumerate(text_a.split(' ')):
            if token in entity_type_dict:
                entity_types_in_text.add(token)
        return entity_types_in_text

    @classmethod
    def _read_tsv(cls, 
                  input_file,
                  quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
                
            return lines


class BlueBERT_GT_Processor(DataProcessor):
    """Processor for the BLUE data set."""

    def __init__(self, 
                 guid_idx           = 0, 
                 text_a_idx         = 1, 
                 in_neighbors_idx   = 2,
                 label_idx          = 3,
                 use_balanced_neg   = False, 
                 no_neg_for_train_dev = False,
                 max_neg_scale      = 2,
                 balanced_neg = ''):
        """If use_balanced_neg is True, you have to assign 
           the 'name of the negative label' to the para 'balanced_neg_label';
           max_neg_scale is used only if use_balanced_neg == True
        """
        
        self.guid_idx = guid_idx
        self.text_a_idx = text_a_idx
        self.in_neighbors_idx = in_neighbors_idx
        self.label_idx = label_idx
        self.use_balanced_neg = use_balanced_neg
        self.no_neg_for_train_dev = no_neg_for_train_dev
        self.max_neg_scale = max_neg_scale
        self.balanced_neg = balanced_neg

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", False)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", False)

    def get_test_examples(self, data_dir, has_header=True):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", has_header)
        
    def get_test_examples_by_file(self, in_test_file, has_header=True):
        """See base class."""
        return self._create_examples(
                self._read_tsv(in_test_file), "test", has_header)
        
    def update_neighbors_by_shortest_path(self, in_neighbors, text_a):
        entity_indices = self.get_entity_indices_by_types(text_a)
        #print('Processing entity_indices', entity_indices)
        new_in_neighbors = graph_algorithm.find_all_shortest_path(
                                                  in_neighbors.split(' '),
                                                  entity_indices)
        #print('new_in_neighbors', new_in_neighbors)
        return ' '.join(new_in_neighbors)


    def update_neighbors_by_surrounding_words(self, in_neighbors):
        new_in_neighbors = graph_algorithm.add_surrounding_words_2_neighbors(
                                                  in_neighbors.split(' '),
                                                  FLAGS.surrounding_words_distance)
        return ' '.join(new_in_neighbors)
    
    def update_neighbors_by_entities(self, in_neighbors, text_a):
        entity_indices = self.get_entity_indices_by_types(text_a)
        new_in_neighbors = graph_algorithm.add_entities_as_neighbors(
                                                  in_neighbors.split(' '),
                                                  entity_indices)
        return ' '.join(new_in_neighbors)

    def _map_label(self, label):
        return label

    def _create_examples(self, 
                         lines, 
                         set_type,
                         has_header=True):
        
        """Creates examples for the training and dev sets."""
        examples = []
        
        neg_scale = 0
        if self.use_balanced_neg and set_type != 'test':
            num_non_neg = 0.
            num_neg = 0.
            for line in lines:
                try:
                    label = self._map_label(tokenization.convert_to_unicode(line[self.label_idx]))
                except IndexError:
                    logging.exception(line)
                    exit(1)
                if label == self.balanced_neg:
                    num_neg += 1.
                else:
                    num_non_neg += 1.
            neg_scale = int(round(num_neg / num_non_neg))
        neg_scale = 1 if neg_scale < 1 else neg_scale
        neg_scale = int(neg_scale)
        
        neg_number = 0
        non_neg_number = 0
        for (i, line) in enumerate(lines):
            # skip header
            if has_header and i == 0:
                continue
            guid = line[self.guid_idx]
            orig_text_a = line[self.text_a_idx]
            in_neighbors = line[self.in_neighbors_idx]
            
            # Sometimes 'orig_text_a' and 'text_a' are slightly different
            text_a = tokenization.convert_to_unicode(orig_text_a)
            
            '''Sometimes 'orig_text_a' and 'text_a' are slightly different, 
               therefore we have to update in_neighbors according to 'text_a' '''
            in_neighbors = tokenization.update_index(
                    orig_text_a, 
                    text_a, 
                    in_neighbors)
            
            if FLAGS.shortest_path_neighbors:
                in_neighbors = self.update_neighbors_by_shortest_path(in_neighbors, text_a)
                
            if FLAGS.surrounding_words_distance > 0:
                in_neighbors = self.update_neighbors_by_surrounding_words(in_neighbors)
        
            if FLAGS.use_entities_as_neighbors:
                in_neighbors = self.update_neighbors_by_entities(in_neighbors, text_a)
            
        
            if set_type == "test":
                # it doesn't matter whether '-1' is label or not, this 'label' will not be used
                label = self.get_labels()[-1]
            else:
                try:
                    label = self._map_label(tokenization.convert_to_unicode(line[self.label_idx]))
                except IndexError:
                    logging.exception(self.label_idx)
                    exit(1)
                    
            if self.use_balanced_neg and set_type == 'train':
                _r = random.randint(1, neg_scale)
                if _r > self.max_neg_scale and label == self.balanced_neg:
                    continue
            
            if self.use_balanced_neg:
                if label == self.balanced_neg:
                    neg_number += 1
                else:
                    non_neg_number += 1
            
            if self.no_neg_for_train_dev and set_type != 'test' and label == self.balanced_neg:
                continue
            examples.append(
                InputExample(
                    guid=guid, 
                    text_a=text_a, 
                    in_neighbors=in_neighbors, 
                    text_b=None, 
                    label=label))
        return examples


class CDRGraphProcessor(BlueBERT_GT_Processor):
    
    def __init__(self,
                 use_balanced_neg = False,
                 no_neg_for_train_dev = False,
                 max_neg_scale    = 2,
                 balanced_neg     = '0'):
        super().__init__(
            text_a_idx       = 5,
            in_neighbors_idx = 6,
            label_idx        = 7,
            use_balanced_neg = use_balanced_neg,
            no_neg_for_train_dev = no_neg_for_train_dev,
            max_neg_scale    = 2,
            balanced_neg     = balanced_neg)
    
    def get_labels(self):
        """See base class."""
        return ["0", "1"]
    
    def get_entity_type_dict(self):
        return {'@ChemicalSrc$':0, '@DiseaseTgt$':1}
    
    def update_neighbors_by_shortest_path(self, in_neighbors, text_a):
        entity_indices = self.get_entity_indices_by_types(text_a)
        _in_neighbors = in_neighbors.split(' ')
        
        if len(text_a.split(' ')) == len(_in_neighbors):
            new_in_neighbors = graph_algorithm.find_all_shortest_path_between(
                                                  _in_neighbors,
                                                  entity_indices['@ChemicalSrc$'],
                                                  entity_indices['@DiseaseTgt$'])
        else:
            print('Cannot not process', text_a)
            return in_neighbors
        return ' '.join(new_in_neighbors)
    
class BioredNoveltyGraphProcessor(BlueBERT_GT_Processor):
    
    def __init__(self,
                 use_balanced_neg = False,
                 no_neg_for_train_dev = False,
                 max_neg_scale    = 2,
                 balanced_neg     = 'None'):
        super().__init__(
            text_a_idx       = 5,
            in_neighbors_idx = 6,
            label_idx        = 8,
            use_balanced_neg = use_balanced_neg,
            no_neg_for_train_dev = no_neg_for_train_dev,
            max_neg_scale    = 2,
            balanced_neg     = balanced_neg)
    
    def get_labels(self):
        """See base class."""
        return ['No', 
                'None', 
                'Novel']
    
    def get_entity_type_dict(self):
        return {'@GeneOrGeneProductSrc$':0, 
                '@DiseaseOrPhenotypicFeatureSrc$':0,
                '@ChemicalEntitySrc$':0,
                '@GeneOrGeneProductTgt$':1,
                '@DiseaseOrPhenotypicFeatureTgt$':1,
                '@ChemicalEntityTgt$':1,}
    
    def update_neighbors_by_shortest_path(self, in_neighbors, text_a):
        entity_indices = self.get_entity_indices_by_types(text_a)        
        entity_types = self.get_entity_types_in_text(text_a)
        src_entity_type = [x for x in entity_types if x.endswith('Src$')][0]
        tgt_entity_type = [x for x in entity_types if x.endswith('Tgt$')][0]
        
        _in_neighbors = in_neighbors.split(' ')
        
        if len(text_a.split(' ')) == len(_in_neighbors):
            new_in_neighbors = graph_algorithm.find_all_shortest_path_between(
                                                  _in_neighbors,
                                                  entity_indices[src_entity_type],
                                                  entity_indices[tgt_entity_type])
        else:
            print('Cannot not process', text_a)
            return in_neighbors
        return ' '.join(new_in_neighbors)
    
class BioredMultiGraphProcessor(BlueBERT_GT_Processor):
    
    def __init__(self,
                 use_balanced_neg = False,
                 no_neg_for_train_dev = False,
                 max_neg_scale    = 2,
                 balanced_neg     = 'None'):
        super().__init__(
            text_a_idx       = 5,
            in_neighbors_idx = 6,
            label_idx        = 7,
            use_balanced_neg = use_balanced_neg,
            no_neg_for_train_dev = no_neg_for_train_dev,
            max_neg_scale    = 2,
            balanced_neg     = balanced_neg)
    
    def get_labels(self):
        """See base class."""
        return ['None', 
                'Association', 
                'Bind',
                'Comparison',
                'Conversion',
                'Cotreatment',
                'Drug_Interaction',
                'Negative_Correlation',
                'Positive_Correlation']
    
    def get_entity_type_dict(self):
        return {'@GeneOrGeneProductSrc$':0, 
                '@DiseaseOrPhenotypicFeatureSrc$':0,
                '@ChemicalEntitySrc$':0,
                '@GeneOrGeneProductTgt$':1,
                '@DiseaseOrPhenotypicFeatureTgt$':1,
                '@ChemicalEntityTgt$':1,}
    
    def update_neighbors_by_shortest_path(self, in_neighbors, text_a):
        entity_indices = self.get_entity_indices_by_types(text_a)        
        entity_types = self.get_entity_types_in_text(text_a)
        src_entity_type = [x for x in entity_types if x.endswith('Src$')][0]
        tgt_entity_type = [x for x in entity_types if x.endswith('Tgt$')][0]
        
        _in_neighbors = in_neighbors.split(' ')
        
        if len(text_a.split(' ')) == len(_in_neighbors):
            new_in_neighbors = graph_algorithm.find_all_shortest_path_between(
                                                  _in_neighbors,
                                                  entity_indices[src_entity_type],
                                                  entity_indices[tgt_entity_type])
        else:
            print('Cannot not process', text_a)
            return in_neighbors
        return ' '.join(new_in_neighbors)
    
class NaryDGVMultiProcessor(BlueBERT_GT_Processor):

    def __init__(self,
                 use_balanced_neg = False,
                 no_neg_for_train_dev = False,
                 max_neg_scale    = 2,
                 balanced_neg     = 'None'):
        super().__init__(
            text_a_idx       = 4,
            in_neighbors_idx = 5,
            label_idx        = 6,
            use_balanced_neg = use_balanced_neg,
            no_neg_for_train_dev = no_neg_for_train_dev,
            max_neg_scale    = max_neg_scale,
            balanced_neg     = balanced_neg)
    
    def get_labels(self):
        """See base class."""
        return ["None", "resistance", "response", "resistance or non-response", "sensitivity"]

    def get_entity_type_dict(self):
        return {'@DRUG$':0, '@GENE$':1, '@VARIANT$':2}


class NaryDVMultiProcessor(BlueBERT_GT_Processor):

    def __init__(self,
                 use_balanced_neg = False,
                 no_neg_for_train_dev = False,
                 max_neg_scale    = 2,
                 balanced_neg     = 'None'):
        super().__init__(
            text_a_idx       = 3,
            in_neighbors_idx = 4,
            label_idx        = 5,
            use_balanced_neg = use_balanced_neg,
            no_neg_for_train_dev = no_neg_for_train_dev,
            max_neg_scale    = max_neg_scale,
            balanced_neg     = balanced_neg)
    
    def get_labels(self):
        """See base class."""
        return ["None", "resistance", "response", "resistance or non-response", "sensitivity"]

    def get_entity_type_dict(self):
        return {'@DRUG$':0, '@VARIANT$':1}
        
class NaryDGVBinaryProcessor(BlueBERT_GT_Processor):

    def __init__(self,
                 use_balanced_neg = False,
                 no_neg_for_train_dev = False,
                 max_neg_scale    = 2,
                 balanced_neg     = 'NO'):
        super().__init__(
            text_a_idx       = 4,
            in_neighbors_idx = 5,
            label_idx        = 6,
            use_balanced_neg = use_balanced_neg,
            no_neg_for_train_dev = no_neg_for_train_dev,
            max_neg_scale    = max_neg_scale,
            balanced_neg     = balanced_neg)
    
    def get_labels(self):
        """See base class."""
        return ["NO", "YES"]

    def get_entity_type_dict(self):
        return {'@DRUG$':0, '@GENE$':1, '@VARIANT$':2}
        
    def _map_label(self, label):
        return 'NO' if label == 'None' else 'YES'
    
class NaryDVBinaryProcessor(BlueBERT_GT_Processor):

    def __init__(self,
                 use_balanced_neg = False,
                 no_neg_for_train_dev = False,
                 max_neg_scale    = 2,
                 balanced_neg     = 'NO'):
        super().__init__(
            text_a_idx       = 3,
            in_neighbors_idx = 4,
            label_idx        = 5,
            use_balanced_neg = use_balanced_neg,
            no_neg_for_train_dev = no_neg_for_train_dev,
            max_neg_scale    = max_neg_scale,
            balanced_neg     = balanced_neg)
  
    def get_labels(self):
        """See base class."""
        return ["NO", "YES"]
        
    def get_entity_type_dict(self):
        return {'@DRUG$':0, '@VARIANT$':1}
    
    def _map_label(self, label):
        return 'NO' if label == 'None' else 'YES'
        
class DDIProcessor(BlueBERT_GT_Processor):

    def __init__(self,
                 use_balanced_neg = False,
                 no_neg_for_train_dev = False,
                 max_neg_scale    = 2,
                 balanced_neg     = 'DDI-false'):
        super().__init__(
            text_a_idx       = 1,
            in_neighbors_idx = 2,
            label_idx        = 3,
            use_balanced_neg = use_balanced_neg,
            no_neg_for_train_dev = no_neg_for_train_dev,
            max_neg_scale    = max_neg_scale,
            balanced_neg     = balanced_neg)
    
    def get_labels(self):
        """See base class."""
        return ["DDI-false", "DDI-mechanism", "DDI-effect", "DDI-advise", "DDI-int"]

    def get_entity_type_dict(self):
        return {'@DRUG$':0}
    
class ChemProtProcessor(BlueBERT_GT_Processor):

    def __init__(self,
                 use_balanced_neg = False,
                 no_neg_for_train_dev = False,
                 max_neg_scale    = 2,
                 balanced_neg     = 'false'):
        super().__init__(
            text_a_idx       = 1,
            in_neighbors_idx = 2,
            label_idx        = 3,
            use_balanced_neg = use_balanced_neg,
            no_neg_for_train_dev = no_neg_for_train_dev,
            max_neg_scale    = max_neg_scale,
            balanced_neg     = balanced_neg)
    
    def get_labels(self):
        """See base class."""
        return ["false", "CPR:4", "CPR:6", "CPR:5", "CPR:9", "CPR:3"]
    
    def get_entity_type_dict(self):
        return {'@GENE$':0, '@CHEMICAL$':1}
    
def convert_single_example(ex_index, 
                           example, 
                           label_list, 
                           entity_type_dict,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        entity_indices = [[0] * FLAGS.max_num_entity_indices for i in range(FLAGS.entity_num)]
        in_neighbors_indices = [[0] * FLAGS.max_num_neighbors for i in range(FLAGS.max_seq_length)]
        
        entity_indices_mask = [[0] * FLAGS.max_num_entity_indices for i in range(FLAGS.entity_num)]
        in_neighbors_indices_mask = [[0] * FLAGS.max_num_neighbors for i in range(FLAGS.max_seq_length)]
        
        input_graph_for_padding_input_example =\
            InputGraph(
                entity_indices,
                in_neighbors_indices,
                entity_indices_mask,
                in_neighbors_indices_mask)
        
        return InputFeatures(
            input_ids=[0] * FLAGS.max_seq_length,
            input_mask=[0] * FLAGS.max_seq_length,
            segment_ids=[0] * FLAGS.max_seq_length,
            input_orig_token_indexes=[0] * FLAGS.max_seq_length,
            input_graph=input_graph_for_padding_input_example,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a, tokens_a_untokenized_indices = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, FLAGS.max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > FLAGS.max_seq_length - 2:
            tokens_a = tokens_a[0:(FLAGS.max_seq_length - 2)]
            tokens_a_untokenized_indices = tokens_a_untokenized_indices[0:(FLAGS.max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    orig_tokens = example.text_a.split(' ')
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    untokenized_index_to_tokenized_indices = {}
  
    for tokenized_index, untokenized_index in enumerate(tokens_a_untokenized_indices):
        if untokenized_index not in untokenized_index_to_tokenized_indices:
            untokenized_index_to_tokenized_indices[untokenized_index] = []
        
        untokenized_index_to_tokenized_indices[untokenized_index].append(tokenized_index + 1)
    
    def gen_new_neighbors_indices(
            untokenized_index_to_tokenized_indices,
            tokens_a_untokenized_indices,
            all_neighbors_indices):
        
        new_neighbors_indices = []
        for token_idx, untokenized_idx in enumerate(tokens_a_untokenized_indices):
            if untokenized_idx >= len(all_neighbors_indices):
                for i in range(len(untokenized_index_to_tokenized_indices)):
                    if i < len(orig_tokens):
                        print(i, orig_tokens[i], untokenized_index_to_tokenized_indices[i])
                    else:
                        print(i, untokenized_index_to_tokenized_indices[i])
                raise('check me')
            untokenized_neighbor_idxs = all_neighbors_indices[untokenized_idx].split('|')
            _new_neighbor_idxs = [token_idx]        
            untokenized_neighbor_idxs.insert(0, str(untokenized_idx))
            for idx in untokenized_neighbor_idxs:
                if int(idx) == -1: # bug?
                    continue
                if not int(idx) in untokenized_index_to_tokenized_indices:
                    continue
                for tokenized_idx in untokenized_index_to_tokenized_indices[int(idx)]:
                    if not tokenized_idx in _new_neighbor_idxs:
                        _new_neighbor_idxs.append(tokenized_idx)
            new_neighbors_indices.append(_new_neighbor_idxs)
        return new_neighbors_indices       
    
    # convert neighbors' indices into WordPiece tokenized indices
    in_neighbors_indices = gen_new_neighbors_indices(
          untokenized_index_to_tokenized_indices,
          tokens_a_untokenized_indices,
          example.in_neighbors.split(' '))
  
    # add neighbors for [CLS] and [SEP]
    in_neighbors_indices.insert(0, [0])
    in_neighbors_indices.append([len(in_neighbors_indices) - 1])
  
    # generate mask for neighbors
    in_neighbors_indices_mask = [[1] * len(in_neighbors_indices[i]) for i in range(len(in_neighbors_indices))]
  
    # extends the token sequence until reaches the max length of the token sequence
    _l = len(in_neighbors_indices)
    while _l < FLAGS.max_seq_length:
        in_neighbors_indices.append([_l])
        in_neighbors_indices_mask.append([0])
        _l += 1
        
    # extends the neighbors of each token until reaches the max num of the neighbors
    for i in range(FLAGS.max_seq_length):
      
        while len(in_neighbors_indices[i]) < FLAGS.max_num_neighbors:
            in_neighbors_indices[i].append(0)
            in_neighbors_indices_mask[i].append(0)
      
        in_neighbors_indices[i] = in_neighbors_indices[i][0:FLAGS.max_num_neighbors]
        in_neighbors_indices_mask[i] = in_neighbors_indices_mask[i][0:FLAGS.max_num_neighbors]
    
        for j in range(FLAGS.max_num_neighbors):
            if in_neighbors_indices[i][j] >= FLAGS.max_seq_length:
                in_neighbors_indices_mask[i][j] = 0
    
  
    # if the token sequence length is larger than the max token sequence length, then discards it
    in_neighbors_indices = in_neighbors_indices[0:FLAGS.max_seq_length]
    in_neighbors_indices_mask = in_neighbors_indices_mask[0:FLAGS.max_seq_length]
     
    entity_indices = [[] for i in range(FLAGS.entity_num)]
  
    num_tokens_a = len(tokens_a)
    # foreach tokenized token i, maps i into untokenized token o,
    #    if untokenized token o is @XXXX$, then add it into corresponding entity_indices
    for tokenized_index in range(1, num_tokens_a):
        orig_index = tokens_a_untokenized_indices[tokenized_index-1]
        orig_token = orig_tokens[orig_index]
        if orig_token in entity_type_dict:
            # discarded later
            try:
                entity_indices[entity_type_dict[orig_token]].append(tokenized_index)
            except:
                print('================>tokens_a:', tokens_a)
                print('================>entity_indices:', entity_indices)
                raise('entity_indices matching failed')
  
    entity_indices_mask = [[1] * len(entity_indices[i]) for i in range(FLAGS.entity_num)]
      
    for i in range(FLAGS.entity_num):
        while len(entity_indices[i]) < FLAGS.max_num_entity_indices:
            entity_indices[i].append(0)
            entity_indices_mask[i].append(0)
        entity_indices[i] = entity_indices[i][0:FLAGS.max_num_entity_indices]
        entity_indices_mask[i] = entity_indices_mask[i][0:FLAGS.max_num_entity_indices]
        
    input_graph = InputGraph(
                          entity_indices,
                          in_neighbors_indices, 
                          entity_indices_mask,
                          in_neighbors_indices_mask)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    #print('tokens', tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < FLAGS.max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == FLAGS.max_seq_length
    assert len(input_mask) == FLAGS.max_seq_length
    assert len(segment_ids) == FLAGS.max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
        for i, x in enumerate(entity_indices):
            tf.logging.info(str("entity_indices: " + str(i) + " = " + str(x)))
        for i, x in enumerate(entity_indices_mask):
            tf.logging.info(str("entity_indices_mask: " + str(i) + " = " + str(x)))
        tf.logging.info("in_neighbors_indices: %s" % " ".join(["|".join(map(str,x)) for x in in_neighbors_indices]))
        tf.logging.info("in_neighbors_indices_mask: %s" % " ".join(["|".join(map(str,x)) for x in in_neighbors_indices_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,        
        input_graph=input_graph,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples,
        label_list,
        entity_type_dict,
        tokenizer,
        output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, 
                                         example, 
                                         label_list,
                                         entity_type_dict, 
                                         tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        
        
        def create_int_feature_for_2d(values):
            out = []
            for indices in values:
                out.extend(indices)
            return create_int_feature(out)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        
        features["entity_indices"] = create_int_feature_for_2d(feature.input_graph.entity_indices)
        features["in_neighbors_indices"] = create_int_feature_for_2d(feature.input_graph.in_neighbors_indices)
        
        features["entity_indices_mask"] = create_int_feature_for_2d(feature.input_graph.entity_indices_mask)
        features["in_neighbors_indices_mask"] = create_int_feature_for_2d(feature.input_graph.in_neighbors_indices_mask)
        
        
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "entity_indices": tf.FixedLenFeature([FLAGS.entity_num, FLAGS.max_num_entity_indices], tf.int64),
        "in_neighbors_indices": tf.FixedLenFeature([seq_length, FLAGS.max_num_neighbors], tf.int64),
        "entity_indices_mask": tf.FixedLenFeature([FLAGS.entity_num, FLAGS.max_num_entity_indices], tf.int64),
        "in_neighbors_indices_mask": tf.FixedLenFeature([seq_length, FLAGS.max_num_neighbors], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def create_model(bert_config, 
                 is_training, 
                 input_ids, 
                 input_mask, 
                 segment_ids,
                 input_graph,
                 labels, 
                 num_labels, 
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertGraphModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        input_graph=input_graph,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()
    entity_states = model.get_entity_states()
    
    hidden_size = output_layer.shape[-1] + entity_states.shape[-1]
      
    output_layer = tf.concat([output_layer, entity_states], axis=-1)
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    
    output_bias = tf.get_variable(
    "output_bias", [num_labels], initializer=tf.zeros_initializer())
    
    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        
    
    return (loss, 
            per_example_loss, 
            logits, 
            probabilities)


def model_fn_builder(bert_config, 
                     num_labels, 
                     init_checkpoint, 
                     learning_rate,
                     num_train_steps, 
                     num_warmup_steps, 
                     use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""
    print('==================>vvvv init_checkpoint', init_checkpoint)
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        
        entity_indices = features["entity_indices"]
        in_neighbors_indices = features["in_neighbors_indices"]    
        
        entity_indices_mask = features["entity_indices_mask"]
        in_neighbors_indices_mask = features["in_neighbors_indices_mask"]
        
        input_graph =\
            InputGraph(
                entity_indices,
                in_neighbors_indices, 
                entity_indices_mask,
                in_neighbors_indices_mask)
        
        is_real_example = None
        
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, 
            is_training,
            input_ids, 
            input_mask,
            segment_ids, 
            input_graph,
            label_ids,
            num_labels, 
            use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            if not os.path.exists(init_checkpoint + '.index'):
                def get_filenames(init_checkpoint):
                    folder_name = os.path.dirname(init_checkpoint)
                    name = os.path.basename(init_checkpoint)
                    return [folder_name + '/' + fname[:-6] for fname in os.listdir(folder_name) if fname.startswith(name) and fname.endswith('.index')]

                def extract_number(filename):
                    import re
                    _filename = os.path.basename(filename)
                    numbers = re.findall(r'\d+',_filename)
                    return (int(numbers[0]) if numbers else -1, filename)

                filename_list = get_filenames(init_checkpoint)
                _init_checkpoint = max(filename_list, key=extract_number)
            else:
                _init_checkpoint = init_checkpoint
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, _init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(_init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(_init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, 
                                 label_list, 
                                 entity_type_dict,
                                 max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, 
                                         example, 
                                         label_list,
                                         entity_type_dict,
                                         max_seq_length, 
                                         tokenizer)

        features.append(feature)
    return features


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "cdr": CDRGraphProcessor,
        "nary_dgv_mul": NaryDGVMultiProcessor,
        "nary_dv_mul": NaryDVMultiProcessor,
        "nary_dgv_bin": NaryDGVBinaryProcessor,
        "nary_dv_bin": NaryDVBinaryProcessor,
        "ddi": DDIProcessor,
        "chemprot": ChemProtProcessor,
        "biored_all_mul": BioredMultiGraphProcessor,
        "biored_novelty": BioredNoveltyGraphProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if bert_config.max_num_neighbors == None:
        bert_config.max_num_neighbors = FLAGS.max_num_neighbors
      
    if bert_config.entity_num == None:
        bert_config.entity_num = FLAGS.entity_num
        
    if bert_config.gt_hidden_size == None:
        bert_config.gt_hidden_size = FLAGS.gt_hidden_size
        
    if bert_config.num_gt_hidden_layers == None:
        bert_config.num_gt_hidden_layers = FLAGS.num_gt_hidden_layers
        
    if bert_config.gt_intermediate_size == None:
        bert_config.gt_intermediate_size = FLAGS.gt_intermediate_size
        
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    bert_config.use_balanced_neg = FLAGS.use_balanced_neg
    bert_config.no_neg_for_train_dev = FLAGS.no_neg_for_train_dev
    bert_config.max_neg_scale = FLAGS.max_neg_scale

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](
            use_balanced_neg = bert_config.use_balanced_neg,
            no_neg_for_train_dev = bert_config.no_neg_for_train_dev)



    label_list = processor.get_labels()
    
    entity_type_dict = processor.get_entity_type_dict()
    entity_type_list = processor.get_entity_type_list()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, 
      entity_type_list=entity_type_list, 
        do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    print('==================>xxxxxxxxxxxx init_checkpoint', FLAGS.init_checkpoint)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, 
            label_list, 
            entity_type_dict, 
            tokenizer, 
            train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(eval_examples) % FLAGS.predict_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(
                eval_examples,
                label_list,                
                entity_type_dict, 
                tokenizer,                
                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.predict(input_fn=eval_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "dev_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Dev results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_eval_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_eval_examples

    if FLAGS.do_predict:
        if FLAGS.test_file != "":
            predict_examples = processor.get_test_examples_by_file(FLAGS.test_file, FLAGS.test_has_header)
        else:
            predict_examples = processor.get_test_examples(FLAGS.data_dir, FLAGS.test_has_header)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(
                predict_examples,
                label_list,                
                entity_type_dict, 
                tokenizer,                
                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

        # result = estimator.evaluate(input_fn=predict_input_fn)
        # tf.logging.info("***** Predict results *****")
        # for key in sorted(result.keys()):
        #     tf.logging.info("  %s = %s", key, str(result[key]))


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

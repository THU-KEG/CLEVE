import csv
import glob
import json
import logging
import os
from typing import List

import tqdm

from transformers import PreTrainedTokenizer
import torch
import random
from collections import defaultdict

import pickle

from transformers.file_utils import add_start_docstrings_to_callable
logger = logging.getLogger(__name__)


negative_weight = 10
sample_type = 'same'


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, tokens, triggerL, triggerR, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.tokens = tokens
        self.triggerL = triggerL
        self.triggerR = triggerR
        self.label = label

class InputContrastExample(object):
    def __init__(self, example_id, tokens, triggerL, triggerR, argL, argR, neg_meta_t, neg_meta_a_t, neg_meta_a_a):
        """Constructs a Input Contrast Example.
        """
        self.example_id = example_id
        self.tokens = tokens
        self.triggerL = triggerL
        self.triggerR = triggerR
        self.argL = argL
        self.argR = argR
        self.neg_meta_t = neg_meta_t
        self.neg_meta_a_t = neg_meta_a_t
        self.neg_meta_a_a = neg_meta_a_a

class InputContrastFeatures(object):
    def __init__(self,example_id,
                input_ids,input_mask,segment_ids,trigger_mask, arg_mask,none_arg_mask, none_arg_length_mask):
        self.example_id = example_id

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.trigger_mask = trigger_mask
        self.arg_mask = arg_mask
        self.none_arg_mask=none_arg_mask
        self.none_arg_length_mask = none_arg_length_mask
    




class InputFeatures(object):
    def __init__(self, example_id, input_ids, input_mask, segment_ids, maskL, maskR, label):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.maskL = maskL
        self.maskR = maskR
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_MI_examples(self,data_dir):
        raise NotImplementedError()

    def _create_examples(self):
        raise NotImplementedError()


class ACEProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        self.train = json.load(open(os.path.join(data_dir,'train.json'),"r"))
        return self._create_examples(self.train, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(json.load(open(os.path.join(data_dir,'dev.json'),"r")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(json.load(open(os.path.join(data_dir,'test.json'),"r")), "test")

    def get_contrast_examples(self,data_dir):
        logger.info("LOOKING AT {} train contrast".format(data_dir))
        lines = pickle.load(open(os.path.join(data_dir,'contrast_examples.pkl'),"rb"))
        pos_meta, neg_meta = self.get_contrast_meta(lines)
        return self._create_contrast_examples(lines, pos_meta, neg_meta, "train-contrast")

    def get_labels(self):
        """See base class."""
        return ['None', 'End-Position', 'Charge-Indict', 'Convict', 'Transfer-Ownership', 'Demonstrate', 'Transport', 'Sentence', 'Appeal', 'Start-Org', 'Start-Position', 'End-Org', 'Phone-Write', 'Nominate', 'Marry', 'Pardon', 'Release-Parole', 'Meet', 'Trial-Hearing', 'Extradite', 'Execute', 'Transfer-Money', 'Elect', 'Injure', 'Acquit', 'Divorce', 'Die', 'Arrest-Jail', 'Declare-Bankruptcy', 'Be-Born', 'Merge-Org', 'Fine', 'Sue', 'Attack']

    # def get_contrast_meta(self,lines):
    #     """
    #     Get meta data for contrastive learning
        
    #     """
    #     positive_meta = {}
    #     negative_meta = {}
    #     # trigger_meta = defaultdict(list)
    #     # line_idx = {}
    #     for example in lines:
    #         if example['event_type']!="None":
    #             metainfo = (example['start'],example['end'],example['file'],example['dir'])
    #             if metainfo not in positive_meta:
                    
    #                 positive_meta[metainfo] = [defaultdict(list),defaultdict(list)]
    #                 negative_meta[metainfo] = [defaultdict(list),defaultdict(list),[]]


    #             trigger_idx = example['trigger_start']
    #             entities = example['entities']
    #             role_idxs = [(e['idx_start'],e['idx_end']) for e in entities if e['role'] != 'None']
    #             none_role_idxs = [(e['idx_start'],e['idx_end']) for e in entities if e['role'] == 'None']
                
    #             # trigger_meta[metainfo].append(trigger_idx)

    #             assert trigger_idx not in positive_meta[metainfo][0]
    #             assert trigger_idx not in negative_meta[metainfo][0]
    #             positive_meta[metainfo][0][trigger_idx].extend(role_idxs)       
    #             negative_meta[metainfo][0][trigger_idx].extend(none_role_idxs)
    #             for role_idx in role_idxs:
    #                 positive_meta[metainfo][1][role_idx].append(trigger_idx)
    #             for none_role_idx in none_role_idxs:
    #                 negative_meta[metainfo][1][none_role_idx].append(trigger_idx)
    #             negative_meta[metainfo][2] = role_idxs+none_role_idxs

    #     return positive_meta,negative_meta

    def get_contrast_meta(self,lines):
        """
        Get meta data for contrastive learning
        
        """
        positive_meta = {}
        negative_meta = {}
        # trigger_meta = defaultdict(list)
        # line_idx = {}
        for example_idx, example in enumerate(lines):
            
            metainfo = example_idx
            if metainfo not in positive_meta:
                
                positive_meta[metainfo] = [defaultdict(list),defaultdict(list)]
                negative_meta[metainfo] = [defaultdict(list),defaultdict(list),[]]

            for node in example['positive_edges'].keys():
                trigger_idx = (node[0],node[1]-1)                                                       #TODO: this part can be cleaner
                entities = example['positive_edges'][node] + example['negative_edges'][node]
                entities = [(e[0],e[1]-1) for e in entities]
                role_idxs = [(e[0],e[1]-1) for e in example['positive_edges'][node]]
                none_role_idxs = [(e[0],e[1]-1) for e in example['negative_edges'][node]]

                assert trigger_idx not in positive_meta[metainfo][0]
                assert trigger_idx not in negative_meta[metainfo][0]
                positive_meta[metainfo][0][trigger_idx].extend(role_idxs)       
                negative_meta[metainfo][0][trigger_idx].extend(none_role_idxs)
                for role_idx in role_idxs:
                    positive_meta[metainfo][1][role_idx].append(trigger_idx)
                for none_role_idx in none_role_idxs:
                    negative_meta[metainfo][1][none_role_idx].append(trigger_idx)
                negative_meta[metainfo][2].extend(role_idxs+none_role_idxs)

        return positive_meta,negative_meta


    # def _create_contrast_examples(self,lines,pos_meta, neg_meta, set_type):
    #     """
    #     Create examples for contrastive training
    #     """

    #     examples = []
    #     for (idx, data_raw) in enumerate(lines):
    #         if data_raw['event_type']!="None":
    #             metainfo = (data_raw['start'],data_raw['end'],data_raw['file'],data_raw['dir'])
    #             for idx2, entity in enumerate(data_raw['entities']):
    #                 if entity['role']!="None":
    #                     e_id = "%s-%s-%s" % (set_type, idx,idx2)
    #                     examples.append(
    #                         InputContrastExample(
    #                             example_id=e_id,
    #                             tokens=data_raw['tokens'],
    #                             triggerL=data_raw['trigger_start'],
    #                             triggerR=data_raw['trigger_end'],
    #                             argL=entity['idx_start'],
    #                             argR=entity['idx_end'],
    #                             neg_meta_t = neg_meta[metainfo][0][data_raw['trigger_start']],
    #                             neg_meta_a_t= neg_meta[metainfo][1][(entity['idx_start'],entity['idx_end'])],
    #                             neg_meta_a_a = [e for e in neg_meta[metainfo][2]]
    #                         )
    #                     )
    #     return examples
    def _create_contrast_examples(self,lines,pos_meta, neg_meta, set_type):
        """
        Create examples for contrastive training
        """

        examples = []
        for (idx, data_raw) in enumerate(lines):
            
            metainfo = idx
            for idx_t, t in enumerate(list(data_raw['positive_edges'].keys())):
                for idx2, entity in enumerate(data_raw['positive_edges'][t]):
                    e_id = "%s-%s-%s-%s" % (set_type, idx, idx_t, idx2)
                    examples.append(
                        InputContrastExample(
                            example_id=e_id,
                            tokens=data_raw['tokens'],
                            triggerL=t[0],
                            triggerR=t[1]-1,
                            argL=entity[0],
                            argR=entity[1]-1,
                            neg_meta_t = neg_meta[metainfo][0][(t[0],t[1]-1)],
                            neg_meta_a_t= neg_meta[metainfo][1][(entity[0],entity[1]-1)],
                            neg_meta_a_a = [e for e in neg_meta[metainfo][2]]
                        )
                    )
        return examples
                         

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (idx, data_raw) in enumerate(lines):
            e_id = "%s-%s" % (set_type, idx)
            examples.append(
                InputExample(
                    example_id=e_id,
                    tokens=data_raw['tokens'],
                    triggerL=data_raw['trigger_start'],
                    triggerR=data_raw['trigger_end']+1,
                    label=data_raw['event_type'],
                )
            )
        return examples

class MAVENProcessor(DataProcessor):
    """Processor for the MAVEN data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(open(os.path.join(data_dir,'train.jsonl'),"r"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(open(os.path.join(data_dir,'valid.jsonl'),"r"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(open(os.path.join(data_dir,'test.jsonl'),"r"), "test")

    def get_labels(self):
        """See base class."""
        return ['None','Change_tool', 'Becoming', 'Using', 'Change_of_leadership', 'Come_together', 'Justifying', 'Name_conferral', 'Collaboration', 'Filling', 'Prison', 'Giving', 'Cause_change_of_position_on_a_scale', 'Limiting', 'Bringing', 'Conquering', 'Forming_relationships', 'Self_motion', 'Commitment', 'Achieve', 'Attack', 'Create_artwork', 'Employment', 'Motion', 'Testing', 'Traveling', 'Preventing_or_letting', 'GiveUp', 'Response', 'Manufacturing', 'Commerce_sell', 'Check', 'Getting', 'Imposing_obligation', 'Earnings_and_losses', 'Assistance', 'Surrounding', 'Resolve_problem', 'Escaping', 'Ratification', 'Presence', 'Warning', 'Exchange', 'Renting', 'Suspicion', 'Agree_or_refuse_to_act', 'Expansion', 'Openness', 'Having_or_lacking_access', 'Recovering', 'Reveal_secret', 'Perception_active', 'Committing_crime', 'Aiming', 'Wearing', 'Creating', 'Writing', 'Hold', 'Education_teaching', 'Incident', 'Quarreling', 'Supply', 'Change', 'Telling', 'Hindering', 'Rewards_and_punishments', 'Protest', 'Know', 'Extradition', 'Departing', 'Sign_agreement', 'Adducing', 'Control', 'Body_movement', 'Releasing', 'Hiding_objects', 'Kidnapping', 'Carry_goods', 'Participation', 'Arrest', 'Sending', 'Reporting', 'Theft', 'Change_sentiment', 'Convincing', 'Preserving', 'Causation', 'Breathing', 'Vocalizations', 'Criminal_investigation', 'Influence', 'Bearing_arms', 'Practice', 'Violence', 'Deciding', 'Being_in_operation', 'Rescuing', 'Temporary_stay', 'Reforming_a_system', 'Catastrophe', 'Besieging', 'Arranging', 'Risk', 'Cause_to_be_included', 'Legal_rulings', 'Confronting_problem', 'Communication', 'Process_start', 'Cure', 'Dispersal', 'Cause_to_amalgamate', 'Institutionalization', 'Competition', 'Killing', 'Submitting_documents', 'Supporting', 'Death', 'Surrendering', 'Recording', 'Revenge', 'Change_event_time', 'Connect', 'Use_firearm', 'Coming_to_be', 'Cause_change_of_strength', 'Ingestion', 'Legality', 'Research', 'Action', 'Award', 'Defending', 'Scrutiny', 'Placing', 'Rite', 'Request', 'Terrorism', 'Process_end', 'Becoming_a_member', 'Expend_resource', 'Commerce_pay', 'Cost', 'Bodily_harm', 'Hostile_encounter', 'GetReady', 'Judgment_communication', 'Containing', 'Robbery', 'Receiving', 'Choosing', 'Lighting', 'Commerce_buy', 'Coming_to_believe', 'Destroying', 'Emptying', 'Building', 'Patrolling', 'Scouring', 'Statement', 'Arriving', 'Social_event', 'Cause_to_make_progress', 'Motion_directional', 'Emergency', 'Labeling', 'Publishing', 'Expressing_publicly', 'Military_operation', 'Removing', 'Damaging']

    def _create_examples(self, fin, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        lines=fin.readlines()
        for (_, data_raw) in enumerate(lines):
            data=json.loads(data_raw)
            for event in data['events']:
                if event['type']=='None of the above':   # This should never happen
                    continue
                for mention in event['mention']:
                    e_id = "%s-%s" % (set_type, mention['id'])
                    examples.append(
                        InputExample(
                            example_id=e_id,
                            tokens=data['content'][mention['sent_id']]['tokens'],
                            triggerL=mention['offset'][0],
                            triggerR=mention['offset'][1],
                            label=event['type'],
                        )
                    )
            for nIns in data['negative_triggers']:
                e_id = "%s-%s" % (set_type, nIns['id'])
                examples.append(
                    InputExample(
                        example_id=e_id,
                        tokens=data['content'][nIns['sent_id']]['tokens'],
                        triggerL=nIns['offset'][0],
                        triggerR=nIns['offset'][1],
                        label='None',
                    )
                )

        return examples


def convert_contrast_examples_to_features(
    examples: List[InputContrastExample],
    max_contrast_ent_per_sent:int,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputContrastFeatures]:

    """
    Loads a data file into a list of `InputFeatures`
    """

    features = []
    def ins(tokens,add):
        add_split = [(e,e[0]) for e in add] + [(e,e[1]+1) for e in add]
        add_split=sorted(add_split,key=lambda x: x[1])
        res=[]
        add_idxs = defaultdict(list)
        for idx,p in enumerate(add_split):
            res.extend(tokenizer.tokenize(" ".join(tokens[(add_split[idx-1][1] if idx>=1 else 0):p[1]])))  # TODO:May occur error when two identical posi..
            add_idxs[p[0]].append(len(res)+1)
        res.extend(tokenizer.tokenize(" ".join(tokens[add_split[-1][1]:])))
        for e in add_idxs:
            assert len(add_idxs[e])==2
        return res,add_idxs

    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert contrast examples to features"):
        trigger_posi = (example.triggerL, example.triggerR)
        arg_posi = (example.argL, example.argR)

        if trigger_posi == arg_posi:
            continue

        entities = list(set(example.neg_meta_a_a))
        neg_meta_t = example.neg_meta_t

        assert arg_posi in entities, ValueError('Contrast processing seems incorrect, please make sure you follow our guide lines and get a correct [nyt_parsed_file]')


        if ex_index % 10000 == 0:
            logger.info("Writing contrast example %d of %d" % (ex_index, len(examples)))
        
        to_rank = list(set([trigger_posi]+entities))
        text, word_idxs = ins(example.tokens,to_rank)

        if max(word_idxs[trigger_posi])>max_length or max(word_idxs[arg_posi])>max_length:
            continue
        
        inputs = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=max_length, return_token_type_ids=True
        )

        if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            logger.info(
                "Attention! you are cropping tokens."
            )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, ValueError('sentence length is not correct, this should never happened.')
        assert len(attention_mask) == max_length, ValueError('sentence length is not correct, this should never happened.')
        assert len(token_type_ids) == max_length, ValueError('sentence length is not correct, this should never happened.')

        trigger_mask = [0]*max_length
        trigger_mask[word_idxs[trigger_posi][0]:word_idxs[trigger_posi][1]] = [1]*(word_idxs[trigger_posi][1]-word_idxs[trigger_posi][0])
        
        assert sum(trigger_mask)!=0, ValueError('This example should have triggers but we found none trigger after processing. This should never happened.')


        arg_mask = [0]*max_length
        arg_mask[word_idxs[arg_posi][0]:word_idxs[arg_posi][1]] = [1]*(word_idxs[arg_posi][1]-word_idxs[arg_posi][0])
        
        assert sum(arg_mask)!=0, ValueError('This example should have args but we found none args after processing. This should never happened.')

        none_arg = [word_idxs[e] for e in neg_meta_t if word_idxs[e][1]<=max_length][:max_contrast_ent_per_sent]
        none_arg_mask = [[0]*max_length]*max_contrast_ent_per_sent
        for idx,e in enumerate(none_arg):
            none_arg_mask[idx][e[0]:e[1]] = [1]*(e[1]-e[0])
            assert sum(none_arg_mask[idx])!=0, ValueError('This example should have entities that are not args but we found less entities that are not args after processing. This should never happened.')

        none_arg_length_mask = [1]*len(none_arg) +[0]*(max_contrast_ent_per_sent-len(none_arg))

        features.append(InputContrastFeatures(example_id=example.example_id, input_ids=input_ids, input_mask=attention_mask, segment_ids=token_type_ids, trigger_mask = trigger_mask, arg_mask = arg_mask, none_arg_mask = none_arg_mask, none_arg_length_mask=none_arg_length_mask))

    return features



def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        textL = tokenizer.tokenize(" ".join(example.tokens[:example.triggerL]))
        textR = tokenizer.tokenize(" ".join(example.tokens[example.triggerL:]))
        maskL = [1.0 for i in range(0,len(textL)+1)] + [0.0 for i in range(0,len(textR)+2)]
        maskR = [0.0 for i in range(0,len(textL)+1)] + [1.0 for i in range(0,len(textR)+2)]
        if len(maskL)>max_length:
            maskL = maskL[:max_length]
        if len(maskR)>max_length:
            maskR = maskR[:max_length]
        inputs = tokenizer.encode_plus(
            textL + ['[unused0]'] + textR, add_special_tokens=True, max_length=max_length, return_token_type_ids=True
        )
        if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            logger.info(
                "Attention! you are cropping tokens."
            )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        assert len(input_ids)==len(maskL)
        assert len(input_ids)==len(maskR)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            maskL = ([0.0] * padding_length) + maskL
            maskR = ([0.0] * padding_length) + maskR
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            maskL = maskL + ([0.0] * padding_length)
            maskR = maskR + ([0.0] * padding_length)

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("example_id: {}".format(example.example_id))
            logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
            logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
            logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
            logger.info("maskL: {}".format(" ".join(map(str, maskL))))
            logger.info("maskR: {}".format(" ".join(map(str, maskR))))
            logger.info("label: {}".format(label))

        features.append(InputFeatures(example_id=example.example_id, input_ids=input_ids, input_mask=attention_mask, segment_ids=token_type_ids, maskL=maskL, maskR=maskR, label=label))

    return features

processors = {"ace": ACEProcessor, "maven": MAVENProcessor}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"ace", 34, "maven", 169}

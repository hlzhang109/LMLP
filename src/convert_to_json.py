import numpy as np
import json
from typing import Optional, List, Tuple, Dict
import re
import collections

import sklearn
import sklearn.metrics

from tabulate import tabulate
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

BaseAtom = collections.namedtuple("Atom", ["predicate", "arguments"])
ps = ['locatedIn', 'neighborOf']
ps_after = ['located in', 'neighbor of']
# planning_lm_id = 'gpt2-large'  # see comments above for all options
# translation_lm_id = 'stsb-roberta-large'  # see comments above for all options
# translation_lm = SentenceTransformer(translation_lm_id)
# tokenizer_plan = AutoTokenizer.from_pretrained(planning_lm_id)
class Atom(BaseAtom):
    def __hash__(self):
        hash_value = hash(self.predicate)
        for a in self.arguments:
            hash_value *= hash(a)
        return hash_value

    def __eq__(self, other):
        return (self.predicate == other.predicate) and (self.arguments == other.arguments)


def trim(string: str) -> str:
    """
    :param string: an input string
    :return: the string without trailing whitespaces
    """
    return re.sub("\A\s+|\s+\Z", "", string)

def parse_rules(rules, delimiter="#####", rule_template=False):
    kb = []
    for rule in rules:
        if rule_template:
            splits = re.split("\A\n?([0-9]?[0-9]+)", rule)
            num = int(splits[1])
            rule = splits[2]
        rule = re.sub(":-", delimiter, rule)
        rule = re.sub("\),", ")"+delimiter, rule)
        rule = [trim(x) for x in rule.split(delimiter)]
        rule = [x for x in rule if x != ""]
        if len(rule) > 0:
            atoms = []
            for atom in rule:
                splits = atom.split("(")
                predicate = splits[0]
                args = [x for x in re.split("\s?,\s?|\)", splits[1]) if x != ""]
                atoms.append(Atom(predicate, args))
            if rule_template:
                kb.append((atoms, num))
            else:
                kb.append(atoms)
    return kb

def load_from_file(path, rule_template=False):
    with open(path, "r") as f:
        text = f.readlines()
        text = [x for x in text if not x.startswith("%") and x.strip() != ""]
        text = "".join(text)
        rules = [x for x in re.split("\.\n|\.\Z", text) if x != "" and
                 x != "\n" and not x.startswith("%")]
        kb = parse_rules(rules, rule_template=rule_template)
        return kb

def convert_rules(path: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open("src/countries/avaliable_rules_r1.json","w") as f_json:
        with open(path, 'rt') as f:
            for line in f.readlines():
                s, p, o = line.split()
                # p = ps_after[0] if p == ps[0] else ps_after[1]
                # s = s.replace('_', ' '); o = o.replace('_', ' ')
                # s = s.replace('-', ' '); o = o.replace('-', ' ')

                # string = '%s %s %s'%(s, p, o) if p == ps_after[1] else '%s is %s %s'%(s, p, o)
                string = '%s %s %s'%(s, p, o)
                triples += [string]
        json.dump(triples,f_json)
    return triples

def convert_examples(path: str) -> List[Tuple[str, str, str]]:
    examples = []
    with open("src/countries/examples_r1.json","w") as f_json:
        with open(path, 'rt') as f:
            for line in f.readlines():
                rules = line.split(',')
                string = ''
                for i in range(len(rules) - 1):
                    p, s, o = rules[i].split()
                    if i == 0 and p == ps[1]:
                        break
                    p = ps_after[0] if p == ps[0] else ps_after[1]
                    s = s.replace('_', ' '); o = o.replace('_', ' ')
                    s = s.replace('-', ' '); o = o.replace('-', ' ')
                    str_ = '%s is the %s %s'%(s, p, o) if p == ps_after[1] else '%s is %s %s'%(s, p, o)
                    if i == 0:
                        string += '%s\n' % (str_)
                    else:
                        string += '%s\n' % (str_)
                else:
                    examples += [string]
        json.dump(examples,f_json)

def convert_examples_r2(path: str) -> List[Tuple[str, str, str]]:
    examples = []
    with open("src/countries/examples_r1.json","w") as f_json:
        with open(path, 'rt') as f:
            for line in f.readlines():
                rules = line.split(',')
                string = ''
                for i in range(len(rules) - 1):
                    if i == 0:
                        p, s, o = rules[i].split()
                        # p = ps_after[0] if p == ps[0] else ps_after[1]
                        # s = s.replace('_', ' '); o = o.replace('_', ' ')
                        # s = s.replace('-', ' '); o = o.replace('-', ' ')
                        # str_ = '%s is the %s %s'%(s, p, o) if p == ps_after[1] else '%s is %s %s'%(s, p, o)
                        # str_ = '%s %s %s'%(s, p, o)
                        # string += '%s\n' % (str_)
                        string += 'Task: %s %s %s\n'%(s, p, o)
                    else:
                        p, s, o = rules[i].split()
                        # p = ps_after[0] if p == ps[0] else ps_after[1]
                        # s = s.replace('_', ' '); o = o.replace('_', ' ')
                        # s = s.replace('-', ' '); o = o.replace('-', ' ')
                        # str_ = '%s is the %s %s'%(s, p, o) if p == ps_after[1] else '%s is %s %s'%(s, p, o)
                        # str_ = '%s %s %s'%(s, p, o)
                        # string += '%s\n' % (str_)
                        string += 'Step %s: %s %s %s\n'%(str(i), s, p, o)
                examples += [string]
        json.dump(examples,f_json)
    
def convert_test_sample(path_countries, path_regions, ground_truth):
    test_countries, trans_tokens, plan_tokens = [], [], []
    with open(path_countries, "r") as f:
        for line in f.readlines():
            line = line[:-1]
            test_countries.append(line)
            # trans_tokens.append(translation_lm.tokenize(line)['input_ids'])
            # plan_tokens.append(tokenizer_plan(line)['input_ids'])
    regions = []
    with open(path_regions, "r") as f:
        for line in f.readlines():
            regions.append(line[:-1])

    ground_truth = load_from_file(ground_truth)

    country2region = {}
    for atom in ground_truth:
        atom = atom[0]
        if atom.predicate == "locatedIn":
            country, region = atom.arguments
            if region in regions:
                country2region[country] = region
    strings = []
    with open("src/countries/test_samples.json","w") as f_json:
        for s in test_countries:
            o = country2region[s]
            # p = ps_after[0]
            # s = s.replace('_', ' '); o = o.replace('_', ' ')
            # s = s.replace('-', ' '); o = o.replace('-', ' ')
            p = ps[0]
            string = 'Task: %s %s %s'%(s, p, o)
            strings += [string]
        json.dump(strings,f_json)
        
        
def txt_to_json(path):
    with open("src/countries/example_r3_train.json","w") as f_json:
        strings = []
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.replace(', ', '\n')
                strings += [line]
convert_rules('src/countries/rules_r1.tsv')
convert_examples_r2('src/countries/examples_r1.tsv')

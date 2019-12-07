# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:03:40 2019

@author: Nehal
"""

import Orange
# Read some data
data = Orange.data.Table('./cn2Auto.csv')

learner = Orange.classification.rules.CN2UnorderedLearner()
#learner = Orange.classification.rules.CN2SDLearner()
#learner = Orange.classification.rules.CN2SDUnorderedLearner()
learner.rule_finder.search_algorithm.beam_width = 10
# construct the learning algorithm and use it to induce a classifier
learner.rule_finder.search_strategy.constrain_continuos = False
learner.rule_finder.general_validator.min_covered_examples = 1000
learner.rule_finder.general_validator.max_rule_length = 10
classifier = learner(data)

for r in classifier.rule_list:
    print (r)
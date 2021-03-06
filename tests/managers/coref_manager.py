import unittest

from derek.data.model import Document, Sentence, Paragraph, Entity, Relation, SortedSpansSet
from derek.coref import CorefTrainer
from tools.common.helper import get_next_props
from tools.coref.coref_trainer import get_strategy_rels


def get_hook(docs_known_rels):
    def evaluate(classifier, epoch):
        pairs_dict = classifier.predict_docs(docs_known_rels, include_probs=True, print_progress=True)
        get_strategy_rels(['pron_rank', 'pron_vote_rank', 'easy_first', 'default'], pairs_dict, docs_known_rels)
        return False

    return evaluate


class TestCorefRunning(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [
            Document('1',
                     ['Во', 'время', 'своих', 'прогулок', 'в', 'окрестностях', 'Симеиза', 'я', 'обратил', 'внимание',
                      'на', 'одинокую', 'дачу', ',', 'стоявшую', 'на', 'крутом', 'склоне', 'горы', '.',
                      'К', 'этой', 'даче', 'не', 'было', 'проведено', 'даже', 'дороги', '.',
                      'Кругом', 'она', 'была', 'обнесена', 'высоким', 'забором', ',', 'с', 'единственной', 'низкой',
                      'калиткой', ',', 'которая', 'всегда', 'была', 'плотно', 'прикрыта', '.'],
                     [Sentence(0, 20), Sentence(20, 29), Sentence(29, 47)],
                     [Paragraph(0, 3)],
                     [Entity('1', 2, 3, 'pron'),
                      Entity('1', 7, 8, 'pron'),
                      Entity('1', 11, 13, 'noun'),
                      Entity('1', 21, 23, 'noun'),
                      Entity('1', 30, 31, 'pron'),
                      Entity('1', 33, 35, 'noun'),
                      Entity('1', 37, 38, 'noun'),
                      Entity('1', 37, 40, 'noun'),
                      Entity('1', 41, 42, 'pron')],
                     {
                         Relation(Entity('1', 2, 3, 'pron'), Entity('1', 7, 8, 'pron'), 'COREF'),
                         Relation(Entity('1', 11, 13, 'noun'), Entity('1', 21, 23, 'noun'), 'COREF'),
                         Relation(Entity('1', 11, 13, 'noun'), Entity('1', 30, 31, 'pron'), 'COREF'),
                         Relation(Entity('1', 21, 23, 'noun'), Entity('1', 30, 31, 'pron'), 'COREF'),
                         Relation(Entity('1', 37, 40, 'noun'), Entity('1', 41, 42, 'pron'), 'COREF'),
                     },
                     {
                         'pos': ['ADP', 'NOUN', 'DET', 'NOUN', 'ADP', 'NOUN', 'PROPN', 'PRON', 'VERB', 'NOUN', 'ADP',
                                 'ADJ', 'NOUN',
                                 'PUNCT', 'VERB', 'ADP', 'ADJ', 'NOUN', 'NOUN', 'PUNCT', 'ADP', 'DET', 'NOUN', 'PART',
                                 'AUX', 'VERB',
                                 'PART', 'NOUN', 'PUNCT', 'ADV', 'PRON', 'AUX', 'VERB', 'ADJ', 'NOUN', 'PUNCT', 'ADP',
                                 'ADJ', 'ADJ',
                                 'NOUN', 'PUNCT', 'PRON', 'ADV', 'AUX', 'ADV', 'VERB', 'PUNCT'],
                         'dt_labels': ['case', 'fixed', 'amod', 'obl', 'case', 'nmod', 'nmod', 'nsubj', 'root', 'obj',
                                       'case',
                                       'amod', 'nmod', 'punct', 'amod', 'case', 'amod', 'obl', 'nmod', 'punct', 'case',
                                       'amod',
                                       'obl', 'advmod', 'aux:pass', 'root', 'advmod', 'nsubj', 'punct', 'advmod',
                                       'nsubj',
                                       'aux:pass', 'root', 'amod', 'obl', 'punct', 'case', 'amod', 'amod', 'conj',
                                       'punct', 'nsubj',
                                       'advmod', 'aux:pass', 'advmod', 'acl:relcl', 'punct'],
                         'dt_head_distances': [3, -1, 1, 5, 1, -2, -1, 1, 0, -1, 2, 1, -3, -1, -2, 2, 1, -3, -1, -1, 2,
                                               1, 3, 2, 1,
                                               0, 1, -2, -1, 3, 2, 1, 0, 1, -2, -1, 3, 2, 1, -5, -1, 4, 3, 2, 1, -11,
                                               -1],
                         'lemmas': ['во', 'время', 'свой', 'прогулка', 'в', 'окрестность', 'Симеиза', 'я', 'обращать',
                                    'внимание',
                                    'на', 'одинокий', 'дача', ',', 'стоять', 'на', 'крутой', 'склон', 'гора', '.', 'к',
                                    'этот',
                                    'дача', 'не', 'быть', 'проводить', 'даже', 'дорога', '.', 'кругом', 'она', 'быть',
                                    'обнесен',
                                    'высокий', 'забор', ',', 'с', 'единственный', 'низкий', 'калитка', ',', 'который',
                                    'всегда',
                                    'быть', 'плотно', 'прикрывать', '.'],
                         'feats': [{}, {'Case': 'Accusative', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                        'Gender': 'Neuter'},
                                   {'Number': 'Plural', 'Pronoun': 'REFLEXIVE', 'Case': 'Genitive'},
                                   {'Case': 'Genitive', 'Animacy': 'Inanimated', 'Number': 'Plural',
                                    'Gender': 'Neuter'}, {},
                                   {'Case': 'Prepositional', 'Animacy': 'Inanimated', 'Number': 'Plural',
                                    'Gender': 'Masculine'},
                                   {'Case': 'Genitive', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Masculine'},
                                   {'Animacy': 'Animated', 'Gender': 'Masculine', 'Number': 'Singular',
                                    'Pronoun': 'DEICTIC',
                                    'Case': 'Nominative'},
                                   {'Number': 'Singular', 'Gender': 'Masculine', 'Tense': 'Past', 'Mode': 'Indicative'},
                                   {'Case': 'Accusative', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Neuter'}, {},
                                   {'Case': 'Accusative', 'Representation': 'Participle', 'Number': 'Singular',
                                    'Gender': 'Feminine',
                                    'Tense': 'NotPast'},
                                   {'Case': 'Accusative', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Feminine'}, {},
                                   {'Case': 'Accusative', 'Representation': 'Participle', 'Number': 'Singular',
                                    'Gender': 'Feminine',
                                    'Tense': 'Past'}, {},
                                   {'Case': 'Prepositional', 'Number': 'Singular', 'Gender': 'Masculine'},
                                   {'Case': 'Prepositional', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Masculine'},
                                   {'Case': 'Genitive', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Feminine'}, {}, {},
                                   {'Case': 'Dative', 'Number': 'Singular', 'Gender': 'Feminine'},
                                   {'Case': 'Dative', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Feminine'}, {},
                                   {'Number': 'Singular', 'Gender': 'Neuter', 'Tense': 'Past', 'Mode': 'Indicative'},
                                   {'Representation': 'Participle', 'Number': 'Singular', 'Gender': 'Neuter',
                                    'Shortness': 'Short',
                                    'Tense': 'Past', 'Voice': 'Passive'}, {},
                                   {'Case': 'Genitive', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Feminine'}, {}, {},
                                   {'Animacy': 'Animated', 'Gender': 'Feminine', 'Number': 'Singular',
                                    'Pronoun': 'PERSONAL',
                                    'Case': 'Nominative'},
                                   {'Number': 'Singular', 'Gender': 'Feminine', 'Tense': 'Past', 'Mode': 'Indicative'},
                                   {'Representation': 'Participle', 'Number': 'Singular', 'Gender': 'Feminine',
                                    'Shortness': 'Short',
                                    'Tense': 'Past', 'Voice': 'Passive'},
                                   {'Case': 'Instrumental', 'Number': 'Singular', 'Gender': 'Masculine'},
                                   {'Case': 'Instrumental', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Masculine'},
                                   {}, {}, {'Case': 'Instrumental', 'Number': 'Singular', 'Gender': 'Feminine'},
                                   {'Case': 'Instrumental', 'Number': 'Singular', 'Gender': 'Feminine'},
                                   {'Case': 'Instrumental', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Feminine'}, {},
                                   {'Case': 'Nominative', 'Number': 'Singular', 'Gender': 'Feminine'}, {},
                                   {'Number': 'Singular', 'Gender': 'Feminine', 'Tense': 'Past', 'Mode': 'Indicative'},
                                   {},
                                   {'Representation': 'Participle', 'Number': 'Singular', 'Gender': 'Feminine',
                                    'Shortness': 'Short',
                                    'Tense': 'Past', 'Voice': 'Passive'}, {}],
                         'said': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O',
                                  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O',
                                  'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                     },
                     {
                         'ne': SortedSpansSet([Entity('1', 6, 7, 'GPE_CITY')])
                     }
                     ),
            Document('1',
                     ['Когда', 'мы', 'шли', 'по', 'тропинке', ',', 'каждый', 'был', 'доволен', 'и', 'думал', ',', 'что',
                      'надул', 'другого', '.',
                      'Петька', 'изредка', 'посапывал', 'носом', '.',
                      'Давно', 'он', 'зарился', 'на', 'моих', 'голубей', ',', 'еще', 'с', 'прошлой', 'зимы', ',', 'а',
                      'теперь', 'вот', 'счастье', 'неожиданно', 'привалило', '.',
                      'А', 'у', 'меня', 'будет', 'пистолет', '.'],
                     [Sentence(0, 16), Sentence(16, 21), Sentence(21, 40), Sentence(40, 46)],
                     [Paragraph(0, 3)],
                     [
                         Entity('1', 1, 2, 'pron'),
                         Entity('1', 16, 17, 'noun'),
                         Entity('1', 22, 23, 'pron'),
                         Entity('1', 25, 26, 'pron'),
                         Entity('1', 25, 27, 'noun'),
                         Entity('1', 42, 43, 'pron'),
                         Entity('1', 44, 45, 'noun'),
                     ],
                     {
                         Relation(Entity('1', 16, 17, 'noun'), Entity('1', 22, 23, 'pron'), 'COREF'),
                         Relation(Entity('1', 25, 26, 'pron'), Entity('1', 42, 43, 'pron'), 'COREF'),
                     },
                     {
                         'pos': ['SCONJ', 'PRON', 'VERB', 'ADP', 'NOUN', 'PUNCT', 'ADJ', 'AUX', 'ADJ', 'CCONJ', 'VERB',
                                 'PUNCT',
                                 'SCONJ', 'VERB', 'ADJ', 'PUNCT', 'NOUN', 'ADV', 'VERB', 'NOUN', 'PUNCT', 'ADV', 'PRON',
                                 'VERB',
                                 'ADP', 'DET', 'NOUN', 'PUNCT', 'ADV', 'ADP', 'NOUN', 'NOUN', 'PUNCT', 'CCONJ', 'ADV',
                                 'PART',
                                 'NOUN', 'ADV', 'VERB', 'PUNCT', 'CCONJ', 'ADP', 'PRON', 'VERB', 'NOUN', 'PUNCT'],
                         'dt_labels': ['mark', 'nsubj', 'advcl', 'case', 'obl', 'punct', 'nsubj', 'cop', 'root', 'cc',
                                       'conj',
                                       'punct', 'mark', 'advcl', 'obj', 'punct', 'nsubj', 'advmod', 'root', 'obl',
                                       'punct', 'advmod',
                                       'nsubj', 'root', 'case', 'amod', 'obl', 'punct', 'advmod', 'case', 'obl', 'nmod',
                                       'punct',
                                       'cc', 'advmod', 'advmod', 'nsubj', 'advmod', 'conj', 'punct', 'cc', 'case',
                                       'root', 'cop',
                                       'nsubj', 'punct'],
                         'dt_head_distances': [8, 1, 6, 1, -2, -1, 2, 1, 0, 1, -2, -1, -2, -3, -1, -1, 2, 1, 0, -1, -1,
                                               2, 1, 0, 2,
                                               1, -3, -1, 2, 1, -7, -1, -1, 5, 4, 1, 2, 1, -15, -1, 2, 1, 0, -1, -2,
                                               -1],
                         'lemmas': ['когда', 'мы', 'идти', 'по', 'тропинка', ',', 'каждый', 'быть', 'довольный', 'и',
                                    'думать', ',',
                                    'что', 'надуть', 'другой', '.', 'Петька', 'изредка', 'посапывать', 'нос', '.',
                                    'давно', 'он',
                                    'зариться', 'на', 'мой', 'голубь', ',', 'еще', 'с', 'прошлый', 'зима', ',', 'а',
                                    'теперь', 'вот',
                                    'счастье', 'неожиданно', 'приваливать', '.', 'а', 'у', 'я', 'быть', 'пистолет',
                                    '.'],
                         'feats': [{}, {'Animacy': 'Animated', 'Number': 'Plural', 'Pronoun': 'DEICTIC',
                                        'Case': 'Nominative'},
                                   {'Number': 'Plural', 'Tense': 'Past', 'Mode': 'Indicative'}, {},
                                   {'Case': 'Dative', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Feminine'}, {},
                                   {'Case': 'Nominative', 'Number': 'Singular', 'Gender': 'Masculine'},
                                   {'Number': 'Singular', 'Gender': 'Masculine', 'Tense': 'Past', 'Mode': 'Indicative'},
                                   {'Number': 'Singular', 'Gender': 'Masculine', 'Shortness': 'Short'}, {},
                                   {'Number': 'Singular', 'Gender': 'Masculine', 'Tense': 'Past', 'Mode': 'Indicative'},
                                   {}, {},
                                   {'Number': 'Singular', 'Gender': 'Masculine', 'Tense': 'Past', 'Mode': 'Indicative'},
                                   {'Case': 'Accusative', 'Animacy': 'Animated', 'Number': 'Singular',
                                    'Gender': 'Masculine'}, {},
                                   {'Case': 'Nominative', 'Animacy': 'Animated', 'Number': 'Singular',
                                    'Gender': 'Masculine'}, {},
                                   {'Number': 'Singular', 'Gender': 'Masculine', 'Tense': 'Past', 'Mode': 'Indicative'},
                                   {'Case': 'Instrumental', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Masculine'},
                                   {}, {},
                                   {'Animacy': 'Animated', 'Gender': 'Masculine', 'Number': 'Singular',
                                    'Pronoun': 'PERSONAL',
                                    'Case': 'Nominative'},
                                   {'Number': 'Singular', 'Gender': 'Masculine', 'Tense': 'Past', 'Mode': 'Indicative'},
                                   {},
                                   {'Animacy': 'Animated', 'Number': 'Plural', 'Pronoun': 'POSSESSIVE',
                                    'Case': 'Accusative'},
                                   {'Case': 'Accusative', 'Animacy': 'Animated', 'Number': 'Plural',
                                    'Gender': 'Masculine'}, {}, {},
                                   {}, {'Case': 'Genitive', 'Number': 'Singular', 'Gender': 'Feminine'},
                                   {'Case': 'Genitive', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Feminine'}, {}, {},
                                   {}, {}, {'Case': 'Nominative', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                            'Gender': 'Neuter'},
                                   {},
                                   {'Number': 'Singular', 'Gender': 'Neuter', 'Tense': 'Past', 'Mode': 'Indicative'},
                                   {}, {}, {},
                                   {'Animacy': 'Animated', 'Gender': 'Masculine', 'Number': 'Singular',
                                    'Pronoun': 'DEICTIC',
                                    'Case': 'Genitive'},
                                   {'Person': 'Third', 'Number': 'Singular', 'Tense': 'NotPast', 'Mode': 'Indicative'},
                                   {'Case': 'Nominative', 'Animacy': 'Inanimated', 'Number': 'Singular',
                                    'Gender': 'Masculine'}, {}],
                         'said': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O',
                                  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O',
                                  'O', 'O', 'O', 'O', 'O', 'O'],
                     },
                     {
                         'ne': SortedSpansSet([Entity('1', 16, 17, 'PERSON')])
                     }
                     )
        ]
        # empty sets are "known" rels
        self.hook = get_hook([doc.without_relations().with_relations(set()) for doc in self.docs])
        self.base_props = {
            "seed": 12345,

            "distance": 10,
            "max_distance": 10,
            "loss": "cross_entropy",
            "optimizer": "momentum",
            "lr_decay": 0.05,
            "momentum": 0.9,
            "dropout": 0.5,
            "internal_size": 10,
            "epoch": 1,
            "batch_size": 64,
            "learning_rate": 0.1,

            "clip_norm": 5,

            "max_candidate_distance": 50,
            "max_entity_distance": 50,
            "max_word_distance": 50,
            "max_sent_distance": 10,
            "max_dt_distance": 10,
            "dist_size": 50,

            "pos_emb_size": 0,
            "morph_feats_emb_size": 0,
            "entities_types_size": 20,

            "morph_feats_size": 0,
            "morph_feats_list": ["Gender", "Animacy", "Number"],

            "encoding_type": "lstm",
            "entity_encoding_size": 10,
            "encoding_size": 10,
            "classifiers": ["exact_match", "intersecting_mentions"],
            "use_filter": False,

            "max_sent_entities_distance": 10,
            "max_token_entities_distance": 20,

            "agreement_types": ["Gender", "Animacy", "Number"],
            "classifier_agreement_size": 0,

            "head_str_match_size": 0,
            "partial_str_match_size": 0,
            "ordered_partial_str_match_size": 0,

            "mention_interrelation_size": 0,
            "mention_distance_size": 0,
            "max_mention_distance": 50,
            "classifier_entity_distance_size": 0,
            "entities_types_in_classifier_size": 0,
            "head_ne_types_size": 0,
            "entities_token_distance_in_classifier_size": 0,
            "entities_sent_distance_in_classifier_size": 0,

            "encoder_entity_types_size": 0,
            "encoder_entity_ne_size": 0,

            "speech_types": ["said"],
            "speech_size": 0,

            "entity_encoding_type": "rnn",

            "classification_dense_size": 20,
        }
        self.experiment_props = {
            "sampling_strategy": ["coref_noun", "coref_pron_cluster", 'coref_pron_cluster_strict', 'coref_pron']
        }

    def test_coref_manager(self):
        for props in get_next_props(self.base_props, self.experiment_props):
            with CorefTrainer(props) as trainer:
                trainer.train(self.docs, self.hook)

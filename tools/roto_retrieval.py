import json
from nltk.translate.bleu_score import sentence_bleu
from pprint import pprint
from multiprocessing import Pool
import munkres
from sklearn.feature_extraction.text import TfidfVectorizer

# ignore_rels = ['HOME_AWAY', 'TEAM_NAME', 'PLAYER_NAME']
from tqdm import tqdm

ignore_rels = []
LARGE_NUM = 10000000


class Record(object):
    def __init__(self, rel, entity, value):
        self.rel = rel
        self.entity = entity
        self.value = value

    def __str__(self):
        return "{}|{}|{}".format(self.rel, self.entity, self.value)


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


class RecordDataset(object):
    def __init__(self, src_file, tgt_file):
        self.records = []
        self.normalized_targets = []
        with open(src_file, encoding='utf-8') as srcf, open(tgt_file, encoding='utf-8') as tgtf:
            idx = 0
            for line_src, line_tgt in zip(srcf, tgtf):
                ls = line_src.strip().split()
                if ls:
                    records = []
                    for triple in ls:
                        value, rel, entity = triple.split('|')
                        try:
                            value = int(value)
                        except ValueError:
                            value = str(value)
                        r = Record(rel, entity, value)
                        records.append(r)
                    self.records.append({'idx': idx, 'records': records, 'target': line_tgt.strip()})
                    idx += 1

        print("Number of records and tgts read: ", len(self.records))
        self.normalize_text()
        self.target_token_set = self.create_token_set()
        self.sklearn_tfidf = TfidfVectorizer(sublinear_tf=True, tokenizer=lambda x: x, preprocessor=lambda x: x,
                                             lowercase=False, token_pattern=None)
        self.tfidf_score = self.sklearn_tfidf.fit_transform(self.normalized_targets).toarray()

    def normalize_text(self):
        for r in self.records:
            tgt_tokens = r['target'].split()
            src_tokens = [str(x.value) for x in r['records']]
            tgt_tokens = self.normalize_sent(tgt_tokens, src_tokens)
            self.normalized_targets.append(tgt_tokens)

    def create_token_set(self):
        token_sets = []
        for sent in self.normalized_targets:
            token_sets.append(set(sent))
        return token_sets

    @staticmethod
    def generalized_jaccard(list_a, list_b):
        keys = set(list_a + list_b)
        counter_a = {}
        counter_b = {}
        for k in keys:
            counter_a[k] = 0
            counter_b[k] = 0
        for x in list_a:
            counter_a[x] += 1
        for x in list_b:
            counter_b[x] += 1
        sum_min = 0
        sum_max = 0
        for k in keys:
            sum_min += min(counter_a[k], counter_b[k])
            sum_max += max(counter_a[k], counter_b[k])
        return sum_min / sum_max

    @staticmethod
    def construct_cost_matrix(query, target):
        cost_matrix = [[LARGE_NUM] * len(target) for _ in range(len(query))]
        for i, q in enumerate(query):
            for j, t in enumerate(target):
                if q.rel == t.rel:
                    if isinstance(q.value, str) and isinstance(t.value, str):
                        if q.value == t.value:
                            cost_matrix[i][j] = 0
                        else:
                            cost_matrix[i][j] = 1
                    elif isinstance(q.value, int) and isinstance(t.value, int):
                        cost_matrix[i][j] = abs(q.value - t.value)
        return cost_matrix

    def calculate_score(self, query_record, target_record):
        query = [r for r in query_record['records'] if r.rel not in ignore_rels]
        target = [r for r in target_record['records'] if r.rel not in ignore_rels]
        query_rels = [r.rel for r in query]
        target_rels = [r.rel for r in target]
        return self.generalized_jaccard(query_rels, target_rels)

    def calculate_alignment(self, a_records, b_records):
        km = munkres.Munkres()
        cost_matrix = self.construct_cost_matrix(a_records, b_records)
        indexes = km.compute(cost_matrix)
        total_cost = 0
        filtered_indexes = []
        for row, column in indexes:
            value = cost_matrix[row][column]
            if value < LARGE_NUM:
                filtered_indexes.append((row, column))
                total_cost += value
        #     print('(%d, %d) -> %d' % (row, column, value))
        # print("total cost: %d" % total_cost)
        return filtered_indexes, total_cost

    def retrieve(self, query_record, topk=3):
        scores = []
        for example in self.records:
            if example['target'] != query_record['target']:
                jaccard = self.calculate_score(query_record, example)
                scores.append((jaccard, example))
        scores.sort(key=lambda x: x[0], reverse=True)
        highest_score = scores[0][0]
        highest_examples = []
        for s in scores:
            if s[0] == highest_score:
                # start calculating the min cost
                filtered_indexes, total_cost = self.calculate_alignment(query_record['records'], s[1]['records'])
                highest_examples.append({'jaccard': s[0], 'min_cost': total_cost, 'alignment': filtered_indexes,
                                         'retrieved': s[1], 'query': query_record})
        highest_examples.sort(key=lambda x: x['min_cost'])
        return highest_examples[0]

    @staticmethod
    def normalize_sent(tgt, src):
        for i in range(len(tgt)):
            if tgt[i] in src:
                try:
                    int(tgt[i])
                    tgt[i] = 'NUMBER'
                except ValueError:
                    tgt[i] = 'TEXT'
        return tgt

    def retrieve_with_target(self, query_record):
        query_tgt_tokens = query_record['target'].split()
        query_src_tokens = [str(x.value) for x in query_record['records']]
        query_tgt_tokens = self.normalize_sent(query_tgt_tokens, query_src_tokens)
        query_token_set = set(query_tgt_tokens)

        scores = []
        for idx, tokens in enumerate(self.normalized_targets):
            if query_record['target'] != self.records[idx]['target']:
                # scores.append((idx, sentence_bleu([query_tgt_tokens], tokens)))
                # scores.append((idx, self.generalized_jaccard(tokens, query_tgt_tokens)))
                score = 0
                for token in tokens:
                    if token in query_token_set:
                        score += self.tfidf_score[idx, self.sklearn_tfidf.vocabulary_[token]]
                scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        filtered_indexes, total_cost = self.calculate_alignment(query_record['records'],
                                                                self.records[scores[0][0]]['records'])
        return {'score': scores[0][1], 'cost': total_cost, 'alignment': filtered_indexes,
                'retrieved': self.records[scores[0][0]], 'query': query_record}


def retrieve(query_record):
    return rd.retrieve(query_record)


def retrieve_with_target(query_record):
    return rd.retrieve_with_target(query_record)


def main():
    global rd
    rd = RecordDataset('../data/rotowire/roto-sent-data.train.src', '../data/rotowire/roto-sent-data.train.tgt')
    print('Start training retrieval')
    with Pool(24) as p:
        retrieved = p.map(retrieve_with_target, rd.records)
    with open('retrieved_target_train.json', 'w', encoding='utf-8') as of:
        json.dump(retrieved, of, cls=MyEncoder)

    print('Start valid retrieval')
    vrd = RecordDataset('../data/rotowire/roto-sent-data.valid.src', '../data/rotowire/roto-sent-data.valid.tgt')
    with Pool(24) as p:
        retrieved = p.map(retrieve_with_target, vrd.records)
    with open('retrieved_target_valid.json', 'w', encoding='utf-8') as of:
        json.dump(retrieved, of, cls=MyEncoder)

    print('Start valid retrieval')
    trd = RecordDataset('../data/rotowire/roto-sent-data.test.src', '../data/rotowire/roto-sent-data.test.tgt')
    with Pool(24) as p:
        retrieved = p.map(retrieve_with_target, trd.records)
    with open('retrieved_target_test.json', 'w', encoding='utf-8') as of:
        json.dump(retrieved, of, cls=MyEncoder)


if __name__ == '__main__':
    main()

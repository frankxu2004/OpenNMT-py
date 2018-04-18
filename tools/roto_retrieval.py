import json
from pprint import pprint
from multiprocessing import Pool
import munkres

# ignore_rels = ['HOME_AWAY', 'TEAM_NAME', 'PLAYER_NAME']
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
                km = munkres.Munkres()
                cost_matrix = self.construct_cost_matrix(query_record['records'], s[1]['records'])
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
                highest_examples.append({'jaccard': s[0], 'min_cost': total_cost, 'alignment': filtered_indexes,
                                         'retrieved': s[1], 'query': query_record})
        highest_examples.sort(key=lambda x: x['min_cost'])
        return highest_examples[0]


def retrieve(query_record):
    return rd.retrieve(query_record)


def main():
    global rd
    rd = RecordDataset('../data/rotowire/roto-sent-data.train.src', '../data/rotowire/roto-sent-data.train.tgt')
    # with Pool(24) as p:
    #     retrieved = p.map(retrieve, rd.records)
    #
    # with open('retrieved_train.json', 'w', encoding='utf-8') as of:
    #     json.dump(retrieved, of, cls=MyEncoder)

    vrd = RecordDataset('../data/rotowire/roto-sent-data.valid.src', '../data/rotowire/roto-sent-data.valid.tgt')
    with Pool(24) as p:
        retrieved = p.map(retrieve, vrd.records)
    with open('retrieved_valid.json', 'w', encoding='utf-8') as of:
        json.dump(retrieved, of, cls=MyEncoder)

    trd = RecordDataset('../data/rotowire/roto-sent-data.test.src', '../data/rotowire/roto-sent-data.test.tgt')
    with Pool(24) as p:
        retrieved = p.map(retrieve, trd.records)
    with open('retrieved_test.json', 'w', encoding='utf-8') as of:
        json.dump(retrieved, of, cls=MyEncoder)


if __name__ == '__main__':
    main()

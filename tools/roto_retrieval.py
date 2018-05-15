import json
from nltk.translate.bleu_score import sentence_bleu
from pprint import pprint
from multiprocessing import Pool
import munkres
from sklearn.feature_extraction.text import TfidfVectorizer
from text2num import text2num, NumberException

# ignore_rels = ['HOME_AWAY', 'TEAM_NAME', 'PLAYER_NAME']
from tqdm import tqdm

ignore_rels = []
LARGE_NUM = 10000000

prons = {"he", "He", "him", "Him", "his", "His", "they", "They", "them", "Them", "their", "Their"}  # leave out "it"
singular_prons = {"he", "He", "him", "Him", "his", "His"}
plural_prons = {"they", "They", "them", "Them", "their", "Their"}

number_words = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
                "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"}

# load all entities
with open("all_ents.json", encoding='utf-8') as f:
    all_ents = set(json.load(f))


def deterministic_resolve(pron, players, teams, cities, curr_ents, prev_ents, max_back=1):
    # we'll just take closest compatible one.
    # first look in current sentence; if there's an antecedent here return None, since
    # we'll catch it anyway
    for j in range(len(curr_ents) - 1, -1, -1):
        if pron in singular_prons and curr_ents[j][2] in players:
            return None
        elif pron in plural_prons and curr_ents[j][2] in teams:
            return None
        elif pron in plural_prons and curr_ents[j][2] in cities:
            return None

    # then look in previous max_back sentences
    if len(prev_ents) > 0:
        for i in range(len(prev_ents) - 1, len(prev_ents) - 1 - max_back, -1):
            for j in range(len(prev_ents[i]) - 1, -1, -1):
                if pron in singular_prons and prev_ents[i][j][2] in players:
                    return prev_ents[i][j]
                elif pron in plural_prons and prev_ents[i][j][2] in teams:
                    return prev_ents[i][j]
                elif pron in plural_prons and prev_ents[i][j][2] in cities:
                    return prev_ents[i][j]
    return None


def extract_entities(sent, all_ents, prons, prev_ents=None, resolve_prons=False,
                     players=None, teams=None, cities=None):
    sent_ents = []
    i = 0
    while i < len(sent):
        if sent[i] in prons:
            if resolve_prons:
                referent = deterministic_resolve(sent[i], players, teams, cities, sent_ents, prev_ents)
                if referent is None:
                    sent_ents.append((i, i + 1, sent[i], True))  # is a pronoun
                else:
                    # print "replacing", sent[i], "with", referent[2], "in", " ".join(sent)
                    sent_ents.append(
                        (i, i + 1, referent[2], False))  # pretend it's not a pron and put in matching string
            else:
                sent_ents.append((i, i + 1, sent[i], True))  # is a pronoun
            i += 1
        elif sent[i] in all_ents:  # findest longest spans; only works if we put in words...
            j = 1
            while i + j <= len(sent) and " ".join(sent[i:i + j]) in all_ents:
                j += 1
            sent_ents.append((i, i + j - 1, " ".join(sent[i:i + j - 1]), False))
            i += j - 1
        else:
            i += 1
    return sent_ents


def annoying_number_word(sent, i):
    ignores = {"three point", "three - point", "three - pt", "three pt", "three - pointer",
               "three - pointers", "three pointers", "three - points"}
    return " ".join(sent[i:i + 3]) not in ignores and " ".join(sent[i:i + 2]) not in ignores


def extract_numbers(sent):
    sent_nums = []
    i = 0
    # print sent
    while i < len(sent):
        toke = sent[i]
        a_number = False
        try:
            itoke = int(toke)
            a_number = True
        except ValueError:
            pass
        if a_number:
            sent_nums.append((i, i + 1, int(toke)))
            i += 1
        elif toke in number_words and annoying_number_word(sent, i):  # get longest span  (this is kind of stupid)
            j = 1
            while i + j <= len(sent) and sent[i + j] in number_words and annoying_number_word(sent, i + j):
                j += 1
            try:
                sent_nums.append((i, i + j, text2num(" ".join(sent[i:i + j]))))
            except NumberException:
                sent_nums.append((i, i + 1, text2num(sent[i])))
            i += j
        else:
            i += 1
    return sent_nums


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
            tgt_tokens = self.normalize_sent(tgt_tokens)
            self.normalized_targets.append(tgt_tokens)

    def create_token_set(self):
        token_sets = []
        for sent in self.normalized_targets:
            token_sets.append(set(sent))
        return token_sets

    @staticmethod
    def normalize_sent(tgt):
        ents = extract_entities(tgt, all_ents, prons)
        nums = extract_numbers(tgt)
        ranges = []
        for ent in ents:
            ranges.append((ent[0], ent[1], 'ENT'))
        for num in nums:
            ranges.append((num[0], num[1], 'NUM'))
        ranges.sort(key=lambda x: x[0])

        masked_sent = []
        i = 0
        while i < len(tgt):
            match = False
            for r in ranges:
                if i == r[0]:
                    match = True
                    masked_sent.append(r[2])
                    i = r[1]
                    break
            if not match:
                masked_sent.append(tgt[i])
                i += 1
        return masked_sent

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

    def retrieve_with_target(self, query_record):
        query_tgt_tokens = query_record['target'].split()
        query_tgt_tokens = self.normalize_sent(query_tgt_tokens)
        query_token_set = set(query_tgt_tokens)

        scores = []
        for idx, tokens in enumerate(self.target_token_set):
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

    def retrieve_with_target_bleu(self, query_record):
        query_tgt_tokens = query_record['target'].split()
        query_tgt_tokens = self.normalize_sent(query_tgt_tokens)

        scores = []
        for idx, tokens in enumerate(self.normalized_targets):
            if query_record['target'] != self.records[idx]['target']:
                scores.append((idx, sentence_bleu([query_tgt_tokens], tokens)))
        scores.sort(key=lambda x: x[1], reverse=True)
        filtered_indexes, total_cost = self.calculate_alignment(query_record['records'],
                                                                self.records[scores[0][0]]['records'])
        return {'score': scores[0][1], 'cost': total_cost, 'alignment': filtered_indexes,
                'retrieved': self.records[scores[0][0]], 'query': query_record}


def retrieve(query_record):
    return rd.retrieve(query_record)


def retrieve_with_target(query_record):
    return rd.retrieve_with_target(query_record)


def retrieve_with_target_bleu(query_record):
    return rd.retrieve_with_target_bleu(query_record)


def main():
    use_target = False
    use_bleu = True
    global rd
    rd = RecordDataset('../data/rotowire/roto-sent-data.train.src', '../data/rotowire/roto-sent-data.train.tgt')
    if not use_target:
        print('Start training retrieval')
        with Pool(24) as p:
            retrieved = p.map(retrieve, rd.records)
        with open('retrieved_train.json', 'w', encoding='utf-8') as of:
            json.dump(retrieved, of, cls=MyEncoder)

        print('Start valid retrieval')
        vrd = RecordDataset('../data/rotowire/roto-sent-data.valid.src', '../data/rotowire/roto-sent-data.valid.tgt')
        with Pool(24) as p:
            retrieved = p.map(retrieve, vrd.records)
        with open('retrieved_valid.json', 'w', encoding='utf-8') as of:
            json.dump(retrieved, of, cls=MyEncoder)

        print('Start valid retrieval')
        trd = RecordDataset('../data/rotowire/roto-sent-data.test.src', '../data/rotowire/roto-sent-data.test.tgt')
        with Pool(24) as p:
            retrieved = p.map(retrieve, trd.records)
        with open('retrieved_test.json', 'w', encoding='utf-8') as of:
            json.dump(retrieved, of, cls=MyEncoder)
    else:
        if not use_bleu:
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

        else:
            print('Start training retrieval')
            with Pool(24) as p:
                retrieved = p.map(retrieve_with_target_bleu, rd.records)
            with open('retrieved_target_bleu_train.json', 'w', encoding='utf-8') as of:
                json.dump(retrieved, of, cls=MyEncoder)

            print('Start valid retrieval')
            vrd = RecordDataset('../data/rotowire/roto-sent-data.valid.src', '../data/rotowire/roto-sent-data.valid.tgt')
            with Pool(24) as p:
                retrieved = p.map(retrieve_with_target_bleu, vrd.records)
            with open('retrieved_target_bleu_valid.json', 'w', encoding='utf-8') as of:
                json.dump(retrieved, of, cls=MyEncoder)

            print('Start valid retrieval')
            trd = RecordDataset('../data/rotowire/roto-sent-data.test.src', '../data/rotowire/roto-sent-data.test.tgt')
            with Pool(24) as p:
                retrieved = p.map(retrieve_with_target_bleu, trd.records)
            with open('retrieved_target_bleu_test.json', 'w', encoding='utf-8') as of:
                json.dump(retrieved, of, cls=MyEncoder)


if __name__ == '__main__':
    main()

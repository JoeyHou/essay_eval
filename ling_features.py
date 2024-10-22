import pandas as pd
import readability

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def get_readability_results(essay):
    return readability.getmeasures(essay, lang='en')

def get_readability_sub_feature(readability_data, feat):
    if feat in [
        'complex_words_dc',
        'characters',
        'long_words'
    ]:
        return readability_data['sentence info'][feat]

def get_number_of_sentences(text):
    return len(sent_tokenize(text))

def load_asap_id(id_dir, split = 'train'):
    id_lst = []
    with open(id_dir + split + '_ids.txt', 'r') as f:
        for line in f.readlines():
            id_lst.append(int(line.strip()))
    return id_lst

if __name__ == '__main__':
    ## Load data ## 
    asap_data_dir = './data/asap/'
    train_ids = load_asap_id(asap_data_dir + '5_splits/fold_0/')
    dev_ids = load_asap_id(asap_data_dir + '5_splits/fold_0/', 'dev')
    test_ids = load_asap_id(asap_data_dir + '5_splits/fold_0/', 'test')
    id2split = dict(zip(train_ids, ['train'] * len(train_ids)))
    id2split.update(dict(zip(dev_ids, ['dev'] * len(dev_ids))))
    id2split.update(dict(zip(test_ids, ['test'] * len(test_ids))))

    # print(len(train_ids), len(dev_ids), len(test_ids))
    asap_data_all = pd.read_excel(asap_data_dir + 'training_set_rel3.xlsx', sheet_name = 'training_set')
    asap_data_all['split'] = asap_data_all.essay_id.apply(lambda x: id2split[x] if x in id2split else 'other')

    ## prepare essay data ##
    asap_data_all = asap_data_all.query('split == "test"')
    essay_batch = list(zip(asap_data_all['essay'].values, asap_data_all['essay_id'].values, asap_data_all['essay_set'].values))
    
    # load existing linguistic features
    existing_hand_crafted = pd.read_csv('data/asap/hand_crafted_v3.csv') ## TODO: change to allow user input
    existing_hand_crafted.index = existing_hand_crafted['item_id']

    # feature to be computed
    features = {
        'hand_crafted': [
            'unique_word',
            'ess_char_len'
        ],
        'readability': [
            "complex_words_dc",
            "long_words",
            "characters"
        ],
        'others': [
            'sentences',
            # 'llema_count',
            # 'syllable_count',
            # 'num_words',
            # 'noun_count',
            # 'stopwords_count',
        ]
    }
    
    processed_feat = []
    
    # existing ones
    for feat in features['hand_crafted']:
        asap_data_all[feat] = asap_data_all['essay_id'].apply(lambda i: existing_hand_crafted.loc[i][feat])
        processed_feat.append(feat)

    # readability
    asap_data_all['readability_data'] = asap_data_all['essay'].apply(get_readability_results)
    asap_data_all['complex_words_dc'] = asap_data_all['readability_data'].apply(lambda x: get_readability_sub_feature(x, 'complex_words_dc'))
    asap_data_all['long_words'] = asap_data_all['readability_data'].apply(lambda x: get_readability_sub_feature(x, 'long_words'))
    asap_data_all['characters'] = asap_data_all['readability_data'].apply(lambda x: get_readability_sub_feature(x, 'characters'))
    processed_feat.extend([
        'complex_words_dc',
        'long_words',
        'characters',
    ])
    
    # others
    asap_data_all['sentences'] = asap_data_all['essay'].apply(get_number_of_sentences)
    processed_feat.append('sentences')

    # find_median
    group_median = asap_data_all[['essay_set'] + processed_feat].groupby('essay_set').agg('median')
    
    for feat in processed_feat:
        asap_data_all[feat + '_median'] = asap_data_all['essay_set'].apply(lambda x: group_median.loc[x][feat])
    processed_feat_medians = [feat + '_median' for feat in processed_feat]
    asap_data_all[['essay_set', 'essay_id'] + processed_feat + processed_feat_medians].to_csv('data/asap/hand_crafted_cleaned.csv', index = False)



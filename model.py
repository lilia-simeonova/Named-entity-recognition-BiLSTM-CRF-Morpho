import codecs
import tensorflow as tf
import numpy as np
import config as cfg

from random import shuffle
from gensim.models.keyedvectors import KeyedVectors
"""
Note: data should end with two new lines
"""
parameters = {}

encoding = cfg.encoding
type_of_emb = 'lstm'
learning_model = 'lstm'

def train_model():
    """
    training a model
    """
    train_const = False
 
    #  ===================================================
    # Those can be run once
    # model = KeyedVectors.load_word2vec_format('embeddings/word2vec.bin', binary=True)
    # model.save_word2vec_format('embeddings/word2vec.txt', binary=False)
    # save_all_words()
    # voc = load_vocabulary(cfg.vocabulary_location)
    # export_trimmed_vectors(voc)
    #
    # =============================================================

    word_vectors = get_trimmed_vectors(cfg.trimmed_embeddings_file)

    training_words, training_tags, _, _, _, training_morf= load_train_vocabulary(
        cfg.train_location)

    training_words_test, training_tags_test, _, _, _, test_morf = load_train_vocabulary(
        cfg.test_location)


    grammar_hash, one_hot_vectors = get_grammar_vectors(training_morf, test_morf)


    # training_pos_ids = get_pos_ids(grammar_hash, training_morf)
    # test_pos_ids = get_pos_ids(grammar_hash, test_morf)

    train_ids = get_pos_ids(grammar_hash, training_morf)
    test_ids = get_pos_ids(grammar_hash, test_morf)

    train_case, test_case = get_case_ids(grammar_hash, training_words, training_words_test)
    train_lex, test_lex = get_lex_ids(grammar_hash, training_words, training_words_test)
    # print(training_pos_ids)

    train_ids['case'] = train_case
    test_ids['case'] = test_case
    train_ids['lex'] = train_lex

    test_ids['lex'] = test_lex
   

    training_data_ids, training_tags_ids, chars_ids, words_to_ids, ids_to_words = build_data(
        training_words, training_tags)

    test_data_ids, test_tags_ids, chars_ids_test, test_id_to_words, _  = build_data(
        training_words_test, training_tags_test)

    placeholders = define_placeholders()

    if train_const:
        parameters = build_model(word_vectors, one_hot_vectors, placeholders)
        sess, saver = initialize_session()
        train(sess, saver, training_data_ids, training_tags_ids, chars_ids,
              test_data_ids, test_tags_ids, chars_ids_test, parameters, train_ids, test_ids, placeholders)
    else:
        parameters = build_model(word_vectors, one_hot_vectors, placeholders)
        sess, saver = initialize_session()
        restore_session(cfg.model, saver, sess)

        evaluate(sess, test_data_ids, test_tags_ids,
                 chars_ids_test, test_ids, placeholders, parameters)

    # run_epoch(sess, saver, training_data_ids, training_tags_ids, test_data_ids,
    #           test_tags_ids, test_id_to_words, parameters, placeholders)
    print('model is trained')


def train(sess, saver, training_data_ids, training_tags_ids, chars_ids, test_data_ids, test_tags_ids, chars_ids_test, parameters, train_ids, test_ids, placeholders):
    best_score = 0
    curr_nepoch_no_imprv = 0
    cfg.nepoch_no_imprv
    cfg.lr
    training_pos_ids = train_ids['pos']
    for epoch in range(cfg.epoch_range):

        print("Epoch Num:", epoch)
        
        # cfg.lr = tf.train.exponential_decay(cfg.lr, global_step, 0.5)
        score = run_epoch(sess, saver, training_data_ids, training_tags_ids, chars_ids,
                          test_data_ids, test_tags_ids, chars_ids_test, parameters, placeholders, train_ids, test_ids)
        # if epoch > 2 and epoch < 6:
        #     cfg.lr = 0.005
        # if epoch < 10 or score < 81:
        # else:
        global_step = 1
        # cfg.lr = cfg.lr * cfg.lr_decay
        # cfg.lr = cfg.lr * cfg.lr_decay 

        cfg.lr = cfg.lr * cfg.lr_decay 
        print('=============================')
        print('current score', score)
        print('best score', best_score)
        if score >= best_score:
            curr_nepoch_no_imprv = 0
            save_session(sess, saver, cfg.model)
            best_score = score
        else:
            print('no improvements, no improvement: ', curr_nepoch_no_imprv)
            curr_nepoch_no_imprv += 1
            # if curr_nepoch_no_imprv > 0:
            #     cfg.lr = cfg.lr * cfg.lr_decay 
            if curr_nepoch_no_imprv >= cfg.nepoch_no_imprv:
                print('no improvements')
                break


def run_epoch(sess, saver, training_data_ids, training_tags_ids, chars_ids, test_data_ids, test_tags_id, 
chars_ids_test, parameters, placeholders, train_ids, test_ids):


    batches_training = create_batches_test(training_data_ids, cfg.batches_size)
    batches_tags= create_batches_test(training_tags_ids, cfg.batches_size)
    batches_chars_ids = create_batches_test(chars_ids, cfg.batches_size)
    batches_pos = create_batches_test(train_ids['pos'], cfg.batches_size)
    batches_pos2 = create_batches_test(train_ids['pos2'], cfg.batches_size)
    batches_pos3 = create_batches_test(train_ids['pos3'], cfg.batches_size)
    batches_pos4 = create_batches_test(train_ids['pos4'], cfg.batches_size)
    batches_pos5 = create_batches_test(train_ids['pos5'], cfg.batches_size)
    batches_case = create_batches_test(train_ids['case'], cfg.batches_size)
    batches_lex = create_batches_test(train_ids['lex'], cfg.batches_size)
    batches_gen = create_batches_test(train_ids['gen'], cfg.batches_size)
    batches_num = create_batches_test(train_ids['num'], cfg.batches_size)
    batches_art = create_batches_test(train_ids['art'], cfg.batches_size)
    
    
    for i, (batch_training, batch_tags, batch_chars_ids, batch_pos, batch_pos2, batch_pos3, batch_pos4, batch_pos5, batch_case, batch_lex,
     batch_gen, batch_num, batch_art) in enumerate(zip(batches_training, batches_tags, batches_chars_ids,
      batches_pos,batches_pos2,batches_pos3,batches_pos4,batches_pos5, batches_case, batches_lex, batches_gen, batches_num, batches_art)):

        batch_ids = {
            'batch_training': batch_training,
            'batch_chars_ids': batch_chars_ids,
            'batch_pos': batch_pos,
            'batch_pos2': batch_pos2,
            'batch_pos3': batch_pos3,
            'batch_pos4': batch_pos4,
            'batch_pos5': batch_pos5,
            'batch_case': batch_case,
            'batch_lex': batch_lex,
            'batch_gen': batch_gen,
            'batch_num': batch_num,
            'batch_art': batch_art,
            'batch_tags': batch_tags
        }

        fd, _ = get_feed(batch_ids, placeholders, cfg.lr, cfg.dropout)
        # break
        _, train_loss = sess.run(
            [parameters['train_op'], parameters['loss']], feed_dict=fd)

    print('train_loss', train_loss)

    f1 = evaluate(sess, test_data_ids, test_tags_id,
                  chars_ids_test, test_ids, placeholders, parameters)
    return f1


def create_batches(training_data, chars_ids, training_tags, n):
    a = [training_data[i:i + n] for i in range(0, len(training_data), n)]
    b = [training_tags[i:i + n] for i in range(0, len(training_tags), n)]
    c = [chars_ids[i:i + n] for i in range(0, len(chars_ids), n)]

    return a, b, c


def create_batches_test(training_data, n):

    return [training_data[i:i + n] for i in range(0, len(training_data), n)]

def get_grammar_vectors(a, b):
    hash_vectors = {}
    ids_to_vectors = {}
    one_hot_vectors = {}
    grammar_hash = {}
    i = 0
    arr = []
    for pos in a:
        for p in pos:
            pp = list(p)
            if len(pp) > 0:
                if hash_vectors.get(pp[0]) is None:
                    hash_vectors[pp[0]] = i
                    ids_to_vectors[i] = pp[0]
                    arr.append(pp[0])
                    i += 1
    for pos in b:
        for p in pos:
            pp = list(p)
            if len(pp) > 0:
                if hash_vectors.get(pp[0] is None):
                    hash_vectors[pp[0]] = i
                    ids_to_vectors[i] = pp[0]
                    i += 1
                    arr.append(pp[0])
    
    # lbl = tf.placeholder(tf.int32,  shape=[None, None])
    # lbl_one_hot = tf.one_hot(lbl, len(arr), 1.0, 0.0)
    

    arr2 = []
    j = 0
    for a in arr:
        arr2.append(j)
        j+= 1

    hash_lex = {
        '-': 0,
        'p': 1,
        'l': 2,
        'o': 3
    }
    hash_pos = {
        '-': 0,
        'N': 1,
        'A': 2,
        'H': 3,
        'P': 4,
        'M': 5,
        'V': 6,
        'D': 7,
        'C': 8,
        'T': 9,
        'R': 10,
        'I': 11
    }
    hash_gender = {
        '-': 0,
        'm': 1,
        'f': 2,
        'n': 3
    }
    hash_number = {
        '-': 0,
        's': 1,
        'p': 2
    }
    hash_art = {
        '-': 0,
        'i': 1,
        'd': 2,
        'h': 3,
        'f': 4
    }
    hash_pos2 = {
        '-': 0,
        'N': 1,
        'A': 1,
        'H': 1,
        'P': 2,
        'M': 2,
        'V': 2,
        'D': 2,
        'C': 2,
        'T': 2,
        'R': 1,
        'I': 2
    }
    hash_pos3 = {
        '-': 0,
        'N': 1,
        'A': 1,
        'H': 1,
        'P': 2,
        'M': 2,
        'V': 2,
        'D': 2,
        'C': 2,
        'T': 2,
        'R': 3,
        'I': 2
    }
    hash_pos4 = {
        '-': 0,
        'N': 1,
        'A': 4,
        'H': 1,
        'P': 2,
        'M': 2,
        'V': 2,
        'D': 2,
        'C': 2,
        'T': 2,
        'R': 3,
        'I': 2
    }
    hash_pos5 = {
        '-': 0,
        'N': 5,
        'A': 4,
        'H': 1,
        'P': 2,
        'M': 2,
        'V': 2,
        'D': 2,
        'C': 2,
        'T': 2,
        'R': 3,
        'I': 2
    }
    hash_case = {
        'u': 0,
        'l': 1
    }
    one_hot_vectors['pos'] = tf.one_hot([0,1,2,3,4,5,6,7,8,9,10,11], 12)
    one_hot_vectors['pos2'] = tf.one_hot([0,1,2], 3)
    one_hot_vectors['pos3'] = tf.one_hot([0,1,2,3], 4)
    one_hot_vectors['pos4'] = tf.one_hot([0,1,2,3,4], 5)
    one_hot_vectors['pos5'] = tf.one_hot([0,1,2,3,4,5], 6)
    one_hot_vectors['gen'] = tf.one_hot([0,1,2,3], 4)
    one_hot_vectors['num'] = tf.one_hot([0,1,2], 3)
    one_hot_vectors['art'] = tf.one_hot([0,1,2,3,4], 5)
    one_hot_vectors['case'] = tf.one_hot([0,1], 2)
    one_hot_vectors['lex'] = tf.one_hot([0,1,2,3], 4)

    grammar_hash['pos_hash'] = hash_pos
    grammar_hash['gen_hash'] = hash_gender
    grammar_hash['num_hash'] = hash_number
    grammar_hash['art_hash'] = hash_art
    grammar_hash['pos2_hash'] = hash_pos2
    grammar_hash['pos3_hash'] = hash_pos3
    grammar_hash['pos4_hash'] = hash_pos4
    grammar_hash['pos5_hash'] = hash_pos5
    grammar_hash['case_hash'] = hash_case
    grammar_hash['lex_hash'] = hash_lex

    return grammar_hash, one_hot_vectors

def get_case_ids(grammar_hash, training_words, test_words):
    case_hash = grammar_hash['case_hash']
    cases = []
    cases_ = []
    for words in training_words:
        case = []
        for word in words:
            if word[0].isupper():
                case.append(case_hash['u'])
            else:
                case.append(case_hash['l'])
        cases.append(case)
    
    for words_ in test_words:
        case_ = []
        for word in words_:
            if word[0].isupper():
                case_.append(case_hash['u'])
            else:
                case_.append(case_hash['l'])
        cases_.append(case_)

    return cases, cases_

def get_lex_ids(grammar_hash, training_words, test_words):
    lex = {}
    lex_hash = grammar_hash['lex_hash']
    lexicons = []
    lexicons_ = []
    for line in codecs.open('./lexicons/pers.txt', 'r', encoding):
        lex[line] = 'p'
    for line in codecs.open('./lexicons/org.txt', 'r', encoding):
        lex[line] = 'o'
    for line in codecs.open('./lexicons/loc.txt', 'r', encoding):
        lex[line] = 'l'
    
    for words in training_words:
        lexicon = []
        for word in words:
            if lex.get(word.lower()):
                lexicon.append(lex_hash[lex[word.lower()]])
            else:
                lexicon.append(lex_hash['-'])
        lexicons.append(lexicon)

    for words_ in test_words:
        lexicon_ = []
        for word_ in words_:
            if lex.get(word_.lower()):
                lexicon_.append(lex_hash[lex[word_.lower()]])
            else:
                lexicon_.append(lex_hash['-'])
        lexicons_.append(lexicon_)

    return lexicons, lexicons_

def get_pos_ids(grammar_hash, training_morf):
    # print(pos_vectors)
    pos_hash = grammar_hash['pos_hash']
    gen_hash = grammar_hash['gen_hash']
    num_hash = grammar_hash['num_hash']
    art_hash = grammar_hash['art_hash']
    pos2_hash = grammar_hash['pos2_hash']
    pos3_hash = grammar_hash['pos3_hash']
    pos4_hash = grammar_hash['pos4_hash']
    pos5_hash = grammar_hash['pos5_hash']

    row = 0
    pos = []
    pos2 = []
    pos3 = []
    pos4 = []
    pos5 = []
    genders = []
    numbers = []
    arts = []
    ids = {}

    for aa in training_morf:
        poses = []
        poses2 = []
        poses3 = []
        poses4 = []
        poses5 = []
        gender = []
        number = []
        art = []
        for a in aa:
            row += 1
            a = list(a)

            if pos_hash.get(a[0]):
                poses.append(pos_hash[a[0]])
            else: 
                poses.append(pos_hash['-'])

            if pos2_hash.get(a[0]):
                poses2.append(pos2_hash[a[0]])
            else: 
                poses2.append(pos2_hash['-'])

            if pos3_hash.get(a[0]):
                poses3.append(pos3_hash[a[0]])
            else: 
                poses3.append(pos3_hash['-'])
            
            if pos4_hash.get(a[0]):
                poses4.append(pos4_hash[a[0]])
            else: 
                poses4.append(pos4_hash['-'])

            if pos5_hash.get(a[0]):
                poses5.append(pos5_hash[a[0]])
            else: 
                poses5.append(pos5_hash['-'])

            if a[0] == 'A' and len(a) > 4:
                # print(a)
                if gen_hash.get(a[1]):
                    gender.append(gen_hash[a[1]])
                else:
                    gender.append(gen_hash['-'])

                if num_hash.get(a[2]):
                    number.append(num_hash[a[2]])
                else:
                    number.append(num_hash['-'])

                if art_hash.get(a[3]):
                    art.append(art_hash[a[3]])
                else:
                    art.append(art_hash['-'])

            elif a[0] == 'N' and len(a) > 4:
                if gen_hash.get(a[2]):
                    gender.append(gen_hash[a[2]])
                else:
                    gender.append(gen_hash['-'])

                if num_hash.get(a[3]):
                    number.append(num_hash[a[3]])
                else:
                    number.append(num_hash['-'])

                if art_hash.get(a[4]):
                    art.append(art_hash[a[4]])
                else:
                    art.append(art_hash['-'])
            elif a[0] == 'H' and len(a) > 3:
                if gen_hash.get(a[1]):
                    gender.append(gen_hash[a[1]])
                else:
                    gender.append(gen_hash['-'])
                if num_hash.get(a[2]):
                    number.append(num_hash[a[2]])
                else:
                    number.append(num_hash['-'])
                if art_hash.get(a[3]):
                    art.append(art_hash[a[3]])
                else:
                    art.append(art_hash['-'])
            elif a[0] == 'P' and len(a) > 9:
                if gen_hash.get(a[7]):
                    gender.append(gen_hash[a[7]])
                else:
                    gender.append(gen_hash['-'])
                if num_hash.get(a[5]):
                    number.append(num_hash[a[5]])
                else:
                    number.append(num_hash['-'])
                if art_hash.get(a[8]):
                    art.append(art_hash[a[8]])
                else:
                    art.append(art_hash['-'])
            elif a[0] == 'M' and len(a) > 5:
                if gen_hash.get(a[2]):
                    gender.append(gen_hash[a[2]])
                else:
                    gender.append(gen_hash['-'])
                if num_hash.get(a[2]):
                    number.append(num_hash[a[2]])
                else:
                    number.append(num_hash['-'])
                if art_hash.get(a[4]):
                    art.append(art_hash[a[4]])
                else:
                    art.append(art_hash['-'])
            else:
                gender.append(gen_hash['-'])
                number.append(num_hash['-'])
                art.append(art_hash['-'])

        genders.append(gender)
        pos.append(poses)
        pos2.append(poses2)
        pos3.append(poses3)
        pos4.append(poses4)
        pos5.append(poses5)
        numbers.append(number)
        arts.append(art)

    ids['pos'] = pos
    ids['pos2'] = pos2
    ids['pos3'] = pos3
    ids['pos4'] = pos4
    ids['pos5'] = pos5
    ids['gen'] = genders
    ids['num'] = numbers
    ids['art'] = arts

    return ids

def get_feed(batch_ids, placeholders, lr=None, dropout=None):

    word_ids_, sequence_lengths_ = pad_sequence(batch_ids['batch_training'])

    char_ids_, word_lengths_ = pad_sequence_chars(batch_ids['batch_chars_ids'])

    batch_pos_, batch_len_ = pad_sequence(batch_ids['batch_pos'])

    batch_pos2_, batch_len2_ = pad_sequence(batch_ids['batch_pos2'])

    batch_pos3_, batch_len3_ = pad_sequence(batch_ids['batch_pos3'])

    batch_pos4_, batch_len4_ = pad_sequence(batch_ids['batch_pos4'])

    batch_pos5_, batch_len5_ = pad_sequence(batch_ids['batch_pos5'])

    batch_case_, batch_case_len_ = pad_sequence(batch_ids['batch_case'])

    batch_lex_, batch_lex_len_ = pad_sequence(batch_ids['batch_lex'])

    batch_gen_, batch_gen_len = pad_sequence(batch_ids['batch_gen'])

    batch_num_, batch_num_len = pad_sequence(batch_ids['batch_num'])
  
    batch_art_, batch_art_len = pad_sequence(batch_ids['batch_art'])

    feed = {
        placeholders['word_ids']: word_ids_,
        placeholders['sequence_lengths']: sequence_lengths_,
        placeholders['char_ids']: char_ids_,
        placeholders['word_lengths']: word_lengths_,
        placeholders['pos_ids']: batch_pos_,
        placeholders['pos2_ids']: batch_pos2_,
        placeholders['pos3_ids']: batch_pos3_,
        placeholders['pos4_ids']: batch_pos4_,
        placeholders['pos5_ids']: batch_pos5_,
        placeholders['case_ids']: batch_case_,
        placeholders['lex_ids']: batch_lex_,
        placeholders['gen_ids']: batch_gen_,
        placeholders['num_ids']: batch_num_,
        placeholders['art_ids']: batch_art_
    }


    if batch_ids['batch_tags'] is not None:

        labels_, _ = pad_sequence(batch_ids['batch_tags'])
        feed[placeholders['labels']] = labels_

    if lr is not None:
        feed[placeholders['lr']] = cfg.lr

    if dropout is not None:
        feed[placeholders['dropout']] = cfg.dropout

    # print (feed)
    return feed, sequence_lengths_


def load_train_vocabulary(path):
 # For english database sent_pos.append(w[-2]) should be w[1]
    sentences = []
    sentence = []
    vocabulary = set()
    vocabulary_for_chars = set()
    training_words = []
    training_tags = []
    tags = []
    voc_tags = set()
    pos_tags = []
    for line in codecs.open(path, 'r', encoding):

        if line == "\n" or line == "\r\n":
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(line)

    for s in sentences:

        sent_words = []
        sent_tags = []
        sent_pos = []
        for word in s:

            w = word.split(' ')

            voc_tags.add(w[-1].rstrip('\r\n'))
            if cfg.replace:
                
                w[-1] = w[-1].rstrip('\r\n')

                if w[-1] == 'I-Loc' or w[-1] == 'I-Pers' or w[-1] == 'I-Org' or w[-1] == 'I-Other':
                    if w[-2][0] != 'R':
                        sent_words.append(w[0].capitalize())

                    else:
                        sent_words.append(w[0])
                else:
                    sent_words.append(w[0])
            else:
                sent_words.append(w[0])
            
            sent_tags.append(w[-1].rstrip('\r\n'))

            if cfg.use_en:
                sent_pos.append(w[1])
            else:
                sent_pos.append(w[-2])
        training_words.append(sent_words)
        training_tags.append(sent_tags)
        pos_tags.append(sent_pos)

    for ws in training_words:
        for wws in ws:
            wws = wws.strip()
            if wws.isdigit():
                wws = "$NUM$"

            else:
                vocabulary.add(wws.lower())
                vocabulary_for_chars.add(wws)

    return training_words, training_tags, vocabulary, voc_tags, vocabulary_for_chars, pos_tags


def get_char_vocab(dataset):

    vocab_char = set()
    for words in dataset:
        for word in words:
            if word != ' ':
                vocab_char.update(word)

    return vocab_char


def load_fasttext_vocab(pretrained_vectors_location):
    vocab = set()
    with open(pretrained_vectors_location, encoding="utf-8") as f:
        for line in f:

            word = line.strip().split(' ')[0]

            vocab.add(word.lower())

    return vocab


def save_vocabulary(vocabulary, location):
    f = codecs.open(location, 'w', encoding)
    for i, word in enumerate(vocabulary):

        f.write(word)
        f.write("\n")


def load_vocabulary(vocabulary_location):
    d = dict()
    i = 0
    for line in codecs.open(vocabulary_location, 'r', encoding):
        line = line.split('\n')[0].encode(encoding)

        d[line] = i

        i += 1

    return d


def load_vocabulary_to_id(vocabulary_location):
    d = dict()
    i = 0
    for line in codecs.open(vocabulary_location, 'r', encoding):
        line = line.split('\n')[0].encode(encoding)

        d[i] = line

        i += 1

    return d


def load_vocabulary_tags(vocabulary_location):
    d = dict()

    i = 0
    for line in codecs.open(vocabulary_location, 'r', encoding):
        line = line.split('\n')[0]

        d[line] = i

        i += 1

    return d


def load_vocabulary_tags_keys(vocabulary_location):
    d = dict()
    i = 0
    for line in codecs.open(vocabulary_location, 'r', encoding):
        line = line.split('\n')[0]

        d[i] = line

        i += 1

    return d


def export_trimmed_vectors(vocab):

    embeddings = np.zeros([len(vocab), cfg.dim])
    with open(cfg.pretrained_vectors_location, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]

            embedding = [float(x) for x in line[1:]]
            if vocab.get(word.encode(encoding)) != None:
                word_idx = vocab[word.encode(encoding)]

                embeddings[word_idx] = np.asarray(embedding)
   
    np.savez_compressed(cfg.trimmed_embeddings_file, embeddings=embeddings)


def get_trimmed_vectors(trimmed_embeddings_file):
    with np.load(trimmed_embeddings_file) as data:

        return data["embeddings"]


def save_all_words():

    _, _, training_words, training_tags, voc_for_chars, _ = load_train_vocabulary(
        cfg.train_location)
    _, _, training_words_test, training_tags_test, _, _ = load_train_vocabulary(
        cfg.test_location)
    _, _, training_words_dev, training_tags_dev, _, _ = load_train_vocabulary(
        cfg.dev_location)

    fast_words = load_fasttext_vocab(cfg.pretrained_vectors_location)
    # print(training_words)
    # comment fast words if nothing else is working...

    vocab_words_ = training_words | training_words_test | training_words_dev

    vocab_words = vocab_words_ & fast_words
    vocab_tags = training_tags | training_tags_test | training_tags_dev
    vocab_words.add("$UNK$")

    vocab_chars = get_char_vocab(voc_for_chars)
    vocab_chars.add("$UNK$")
    save_vocabulary(vocab_words, cfg.vocabulary_location)
    save_vocabulary(vocab_tags, cfg.vocabulary_tags_location)
    save_vocabulary(vocab_chars, cfg.vocabulary_chars_location)


def build_model(wordVectors, one_hot_vectors, placeholders):

    word_embeddings = add_embeddings(wordVectors, placeholders, one_hot_vectors)

    parameters['logits'] = add_logits_op(word_embeddings, placeholders)
    parameters['labels_pred'] = add_pred_op(parameters['logits'])
    #  Operations, remove them from parameters and put them in operations
    parameters['loss'], parameters['trans_params'] = add_loss_op(
        parameters['logits'], placeholders)
    parameters['train_op'] = add_train_op(
        cfg._lr_m, cfg.lr, parameters['loss'], placeholders)

    print('model is finally built')
    return parameters


def define_placeholders():
    word_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                          name="word_ids")
    pos_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                          name="pos2_ids")
    pos2_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                          name="pos_ids")
    pos3_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                          name="pos_ids")
    pos4_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                          name="pos_ids")
    pos5_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                          name="pos_ids")  
    case_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                          name="case_ids")  
    lex_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                          name="lex_ids")                                                                                  
    gen_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                          name="gen_ids")
    num_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                          name="num_ids")
    art_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                          name="art_ids")
    labels_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                        name="labels")
    sequence_lengths_placeholder = tf.placeholder(tf.int32, shape=[None],
                                                  name="sequence_lengths")
    dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=[],
                                         name="dropout")
    lr_learning_rate = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="lr")
    word_lengths_placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                              name="word_lengths")

    char_ids_placeholder = tf.placeholder(tf.int32, shape=[None, None, None], name = "chars_ids")

    flag = tf.placeholder(tf.int32, name="flag")
    placeholders = {
        'labels': labels_placeholder,
        'word_ids': word_ids_placeholder,
        'pos_ids': pos_ids_placeholder,
        'pos2_ids': pos2_ids_placeholder,
        'pos3_ids': pos3_ids_placeholder,
        'pos4_ids': pos4_ids_placeholder,
        'pos5_ids': pos5_ids_placeholder,
        'case_ids': case_ids_placeholder,
        'lex_ids': lex_ids_placeholder,
        'gen_ids': gen_ids_placeholder,
        'num_ids': num_ids_placeholder,
        'art_ids': art_ids_placeholder,
        'word_lengths': word_lengths_placeholder,
        'sequence_lengths': sequence_lengths_placeholder,
        'char_ids': char_ids_placeholder,
        'dropout': dropout_placeholder,
        'lr': lr_learning_rate,
        'flag': flag
    }
    return placeholders


def add_embeddings(wordVectors, placeholders, one_hot_vectors):

    nchars = len(load_vocabulary(cfg.vocabulary_chars_location))
    nwords = len(load_vocabulary(cfg.vocabulary_location))

    pos_vectors = one_hot_vectors['pos']
    pos2_vectors = one_hot_vectors['pos2']
    pos3_vectors = one_hot_vectors['pos3']
    pos4_vectors = one_hot_vectors['pos4']
    pos5_vectors = one_hot_vectors['pos5']
    case_vectors = one_hot_vectors['case']
    lex_vectors = one_hot_vectors['lex']
    gen_vectors = one_hot_vectors['gen']
    num_vectors = one_hot_vectors['num']
    art_vectors = one_hot_vectors['art']



    if cfg.use_words == True:

        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(
                wordVectors,
                name="_word_embeddings",
                dtype=tf.float32,
                trainable=False)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                    placeholders['word_ids'], name="word_embeddings")
    
    if type_of_emb == 'lstm':
        with tf.variable_scope("chars"):

            _char_embeddings = tf.get_variable(
                name="_char_embeddings",
                dtype=tf.float32,
                shape=[nchars, cfg.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                    placeholders['char_ids'], name="char_embeddings")

            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings,
                                        shape=[s[0] * s[1], s[-2], cfg.dim_char])
            word_lengths = tf.reshape(
                placeholders['word_lengths'], shape=[s[0] * s[1]])

            cell_fw = tf.contrib.rnn.LSTMCell(cfg.hidden_size_char,
                                            state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(cfg.hidden_size_char,
                                            state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, char_embeddings,
                sequence_length=word_lengths, dtype=tf.float32)

            # read and concat output
            _, ((_, output_fw), (_, output_bw)) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.reshape(output,
                                shape=[s[0], s[1], 2 * cfg.hidden_size_char])
            # only feed lstm on chars level
            # output = output_bw
            # shape = (batch size, max sentence length, char hidden size)
            # output = tf.reshape(output,
                                # shape=[s[0], s[1], cfg.hidden_size_char])
            if cfg.use_chars == True:
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        if cfg.use_pos:
        
            with tf.variable_scope("pos"):
                _pos_embeddings = tf.Variable(
                    pos_vectors,
                    name="_pos_embeddings",
                    dtype=tf.float32,
                    trainable=False)
                pos_embeddings = tf.nn.embedding_lookup(_pos_embeddings,
                                                        placeholders['pos_ids'], name="pos_embedding")

            word_embeddings = tf.concat([word_embeddings, pos_embeddings], axis=-1)
        if cfg.use_pos2:
            with tf.variable_scope("pos2"):
                _pos2_embeddings = tf.Variable(
                    pos2_vectors,
                    name="_pos2_embeddings",
                    dtype=tf.float32,
                    trainable=False)
                pos2_embeddings = tf.nn.embedding_lookup(_pos2_embeddings,
                                                        placeholders['pos2_ids'], name="pos2_embedding")

            word_embeddings = tf.concat([word_embeddings, pos2_embeddings], axis=-1)
        if cfg.use_pos3:
            with tf.variable_scope("pos3"):
                _pos3_embeddings = tf.Variable(
                    pos3_vectors,
                    name="_pos3_embeddings",
                    dtype=tf.float32,
                    trainable=False)
                pos3_embeddings = tf.nn.embedding_lookup(_pos3_embeddings,
                                                        placeholders['pos3_ids'], name="pos3_embedding")

            word_embeddings = tf.concat([word_embeddings, pos3_embeddings], axis=-1)
        if cfg.use_pos4:
            with tf.variable_scope("pos4"):
                _pos4_embeddings = tf.Variable(
                    pos4_vectors,
                    name="_pos4_embeddings",
                    dtype=tf.float32,
                    trainable=False)
                pos4_embeddings = tf.nn.embedding_lookup(_pos4_embeddings,
                                                        placeholders['pos4_ids'], name="pos4_embedding")

            word_embeddings = tf.concat([word_embeddings, pos4_embeddings], axis=-1)
        if cfg.use_pos5:
            with tf.variable_scope("pos5"):
                _pos5_embeddings = tf.Variable(
                    pos5_vectors,
                    name="_pos5_embeddings",
                    dtype=tf.float32,
                    trainable=False)
                pos5_embeddings = tf.nn.embedding_lookup(_pos5_embeddings,
                                                        placeholders['pos5_ids'], name="pos5_embedding")

            word_embeddings = tf.concat([word_embeddings, pos5_embeddings], axis=-1)
        if cfg.use_case:
            with tf.variable_scope("case"):
                _case_embeddings = tf.Variable(
                    case_vectors,
                    name="_case_embeddings",
                    dtype=tf.float32,
                    trainable=False)
                case_embeddings = tf.nn.embedding_lookup(_case_embeddings,
                                                        placeholders['case_ids'], name="case_embedding")

            word_embeddings = tf.concat([word_embeddings, case_embeddings], axis=-1)

        if cfg.use_lex:
            with tf.variable_scope("lex"):
                _lex_embeddings = tf.Variable(
                    lex_vectors,
                    name="_lex_embeddings",
                    dtype=tf.float32,
                    trainable=False)
                lex_embeddings = tf.nn.embedding_lookup(_lex_embeddings,
                                                        placeholders['lex_ids'], name="lex_embedding")


            word_embeddings = tf.concat([word_embeddings, lex_embeddings], axis=-1)

        if cfg.use_morf:  
            if cfg.use_gen:
                with tf.variable_scope("gen"):
                    _gen_embeddings = tf.Variable(
                        gen_vectors,
                        name="_gen_embeddings",
                        dtype=tf.float32,
                        trainable=False)
                    gen_embeddings = tf.nn.embedding_lookup(_gen_embeddings,
                                                            placeholders['gen_ids'], name="gen_embedding")
                word_embeddings = tf.concat([word_embeddings, gen_embeddings], axis=-1)
            if cfg.use_num:
                with tf.variable_scope("num"):
                    _num_embeddings = tf.Variable(
                        num_vectors,
                        name="_num_embeddings",
                        dtype=tf.float32,
                        trainable=False)
                    num_embeddings = tf.nn.embedding_lookup(_num_embeddings,
                                                            placeholders['num_ids'], name="num_embedding")
                word_embeddings = tf.concat([word_embeddings, num_embeddings], axis=-1)
            if cfg.use_art:    
                with tf.variable_scope("art"):
                    _art_embeddings = tf.Variable(
                        art_vectors,
                        name="_art_embeddings",
                        dtype=tf.float32,
                        trainable=False)
                    art_embeddings = tf.nn.embedding_lookup(_art_embeddings,
                                                            placeholders['art_ids'], name="art_embedding")

                word_embeddings = tf.concat([word_embeddings, art_embeddings], axis=-1)



    word_embeddings_ = tf.nn.dropout(word_embeddings, placeholders['dropout'])

    return word_embeddings_

    if type_of_emb == 'cnn':
       
        token_lengths = tf.placeholder(tf.int32, [None, None], name="tok_lengths")
        input_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="input_dropout_keep_prob")
        with tf.variable_scope("char-forward", reuse=False):
            _char_embeddings = tf.get_variable(
                name="_char_embeddings",
                dtype=tf.float32,
                shape=[nchars, cfg.dim_char])
            char_embeddings_lookup = tf.nn.embedding_lookup(_char_embeddings, placeholders['char_ids'])
            s = tf.shape(char_embeddings_lookup)
            max_token_len =  tf.reshape(
                    placeholders['word_lengths'], shape=[s[0] * s[1]])


            char_embeddings_flat = tf.reshape(char_embeddings_lookup, 
            tf.stack([cfg.batches_size*placeholders['sequence_lengths'], max_token_len, cfg.dim]))

            input_feats_expanded_drop = tf.nn.dropout(char_embeddings_lookup, input_dropout_keep_prob)


            with tf.name_scope("char-cnn"):
                filter_shape = [1, 0, 100, 100]

                w = tf.get_variable("conv0_w", initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                # w = tf.get_variable("conv0_w", shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("conv0_b", initializer=tf.constant(0.01, shape=[100]))
                conv0 = tf.nn.conv2d(input_feats_expanded_drop, w, strides=[1, 1, 1, 1], padding="SAME", name="conv0")
                print("conv0", conv0.get_shape())
                h_squeeze = tf.squeeze(conv0, [1])
                print("squeeze", h_squeeze.get_shape())
                hidden_outputs = tf.reduce_max(h_squeeze, 1)
                print("max", hidden_outputs.get_shape())
                # hidden_outputs = tf.reshape(hidden_outputs, tf.stack([cfg.batches_size, placeholders['sequence_lengths'], 100]))
                # word_embeddings = tf.concat([word_embeddings, hidden_outputs], axis=-1)
        return word_embeddings

    if type_of_emb == 'att':
        with tf.variable_scope("chars"):

            _char_embeddings = tf.get_variable(
                name="_char_embeddings",
                dtype=tf.float32,
                shape=[nchars, cfg.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                    placeholders['char_ids'], name="char_embeddings")

            # put the time dimension on axis=1
            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings,
                                        shape=[s[0] * s[1], s[-2], cfg.dim_char])
            word_lengths = tf.reshape(
                placeholders['word_lengths'], shape=[s[0] * s[1]])

            # bi lstm on chars
            cell_fw = tf.contrib.rnn.LSTMCell(cfg.hidden_size_char,
                                            state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(cfg.hidden_size_char,
                                            state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, char_embeddings,
                sequence_length=word_lengths, dtype=tf.float32)

            # read and concat output
            _, ((_, output_fw), (_, output_bw)) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)

            # shape = (batch size, max sentence length, char hidden size)
            output = tf.reshape(output,
                                shape=[s[0], s[1], 2 * cfg.hidden_size_char])

            # Output is the chars representation
            # ===========================================================


            with tf.variable_scope("attention"):
                
                #         W = tf.get_variable("W", dtype=tf.float32,shape=[2 * cfg.hidden_size_lstm, cfg.ntags])
                # initializer=tf.zeros_initializer()

                # Working solution
                # ==============================
                # weights_words = tf.Variable(tf.random_normal([ cfg.batches_size,   cfg.dim , cfg.hidden_size_lstm] ))
                # weights_chars = tf.Variable(tf.random_normal([cfg.batches_size, 2 * cfg.dim_char, 2 * cfg.hidden_size_char ]))
                # weights_all = tf.Variable(tf.random_normal([cfg.batches_size,  2* cfg.dim_char + cfg.dim,  2 * cfg.hidden_size_char + cfg.hidden_size_lstm]))
                # ==============================
                # Working solution
                # weights_words = tf.Variable(tf.random_normal([ cfg.batches_size,   cfg.dim , cfg.hidden_size_lstm] ))
                # weights_chars = tf.Variable(tf.random_normal([cfg.batches_size, 2 *cfg.dim_char, cfg.hidden_size_char ]))
                # weights_all = tf.Variable(tf.random_normal([cfg.batches_size,  cfg.dim + cfg.dim_char,  cfg.hidden_size_lstm]))
               

                weights_words = tf.Variable(tf.random_normal([ cfg.batches_size,  cfg.hidden_size_lstm, cfg.dim ] ))
                b1 = tf.get_variable('b1',[1,],
                         initializer=tf.constant_initializer(0.0))
                b2 = tf.get_variable('b2',[1,],
                         initializer=tf.constant_initializer(0.0))

                weights_chars = tf.Variable(tf.random_normal([cfg.batches_size, 2 *cfg.hidden_size_char, 2 *cfg.dim_char ]))
                weights_all = tf.Variable(tf.random_normal([cfg.batches_size,   cfg.hidden_size_lstm, cfg.dim]))
               

                # weights_words = tf.Variable(tf.random_normal([ cfg.batches_size,  cfg.hidden_size_lstm,  cfg.dim ]))
                # weights_chars = tf.Variable(tf.random_normal([cfg.batches_size,  2 * cfg.hidden_size_char,2 * cfg.dim_char]))
                # weights_all = tf.Variable(tf.random_normal([cfg.batches_size,   2 * cfg.hidden_size_char + cfg.hidden_size_lstm, 2 * cfg.dim_char + cfg.dim]))
            
                # word_embeddings = tf.concat([word_embeddings, output], axis=-1)

                h1    = tf.matmul(word_embeddings, weights_words)  # The \sigma function

                h2    = tf.matmul( output, weights_chars)
                
                             

                res = tf.nn.tanh(tf.add(h1, h2))
  
                h3 = tf.nn.sigmoid(tf.matmul(res, weights_all))

                one_vector = tf.ones_like(h3)
                h3_reverse = tf.subtract(h3, one_vector, name=None)
                # print('h3 reverse shape', h3_reverse.shape)

                # h3 = tf.reshape(h3, shape = [h3.shape[0], h3.shape[1], -1])
                # h3 = tf.transpose(h3, perm=[0, 1,2])

                # sum1 = tf.einsum('aij,aik->ajk', word_embeddings, h3)
                sum1 = tf.multiply(word_embeddings, h3)






                # print('sum1', sum1.shape)
                sum2 = tf.multiply(output, h3_reverse)
                # print('sum2', sum2.shape)
                res2 = tf.add(sum1, sum2)
                # print('a:', a)
                # h3 = tf.nn.tanh(tf.matmul(res, weights_all))
                s = 1 - tf.losses.cosine_distance(word_embeddings, output, dim=0)

                
                # print('shape h3', h3.shape)
                # Current issue is with dropout of h3!!!!!!!!!!!!!!!
                # print('@@@@@@@@@@@@@@@@@@@@',aa)
                # print('shape of h3', h3)

        # To test: 
        # tan sig ---- 78.9
        # sig sig ---- 78.26
        # res2 concat instead of add ---- 80.68
        # use multiply instead of matmul ----
        # add another sigmoid layer on each level ----
        word_embeddings_ = tf.nn.dropout(res2, placeholders['dropout'])
        
        return word_embeddings_

def add_logits_op(word_embeddings, placeholders):
    """Defines logits

    For each word in each sentence of the batch, it corresponds to a vector
    of scores, of dimension equal to the number of tags.
    """

    if learning_model == "lstm":
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(cfg.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(cfg.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, word_embeddings,
                sequence_length=placeholders['sequence_lengths'], dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            # print('lstm output shape', output.shape)
            output = tf.nn.dropout(output, placeholders['dropout'])

        with tf.variable_scope("proj"):

            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * cfg.hidden_size_lstm, cfg.ntags])

            b = tf.get_variable("b", shape=[cfg.ntags],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * cfg.hidden_size_lstm])
            print('=============')

            pred = tf.matmul(output, W) + b
            logits = tf.reshape(pred, [-1, nsteps, cfg.ntags])
        return logits

    if learning_model == 'cnn':
        print ('a')


def add_pred_op(logits):
    """

    This op is defined only in the case where we don't use a CRF since in
    that case we can make the prediction "in the graph" (thanks to tf
    functions in other words). With theCRF, as the inference is coded
    in python and not in pure tensroflow, we have to make the prediciton
    outside the graph.
    """
    labels_pred = tf.cast(tf.argmax(logits, axis=-1),
                          tf.int32)
    return labels_pred


def add_loss_op(logits,  placeholders):
    """Defines the loss"""
    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
        logits, placeholders['labels'], placeholders['sequence_lengths'])
    trans_params = trans_params  # need to evaluate it for decoding
    loss = tf.reduce_mean(-log_likelihood) 

    return loss, trans_params


def add_train_op(_lr_m, lr, loss, placeholders):
    with tf.variable_scope("train_step"):
        if _lr_m == 'adam':
            optimizer = tf.train.AdamOptimizer(placeholders['lr'])
        elif _lr_m == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(placeholders['lr'])
        elif _lr_m == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(placeholders['lr'], 0.95, 1e-6)
        elif _lr_m == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(placeholders['lr'])
        elif _lr_m == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=placeholders['lr'], momentum=0.9, use_nesterov=False)
        elif _lr_m == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(placeholders['lr'])
        elif _lr_m == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(placeholders['lr'], learning_rate_power=-0.5,
    initial_accumulator_value=0.1,
    l1_regularization_strength=0.0,
    l2_regularization_strength=0.0,
    use_locking=False)
        else:
            raise NotImplementedError("Unknown method {}".format(_lr_m))
        if cfg.clip > 0:
            grads, vs = zip(*optimizer.compute_gradients(loss))
            grads, gnorm = tf.clip_by_global_norm(grads, cfg.clip)
            train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            train_op = optimizer.minimize(loss)

    return train_op


def initialize_session():
    """Defines sess and initialize the variables"""
    # logger.info("Initializing tf session")
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    return sess, saver


def pad_sequence(training_data):
    sequence_padded = []
    sequence_length = []

    max_length = max(map(lambda x: len(x), training_data))
    # max_length = 20
    for data in training_data:
        data = list(data)
        data_ = data[:max_length] + [0] * max(max_length - len(data), 0)
        sequence_padded += [data_]
        sequence_length += [min(len(data), max_length)]
    # print(sequence_length)
    return sequence_padded, sequence_length


def pad_sequence_chars(sequences):

    sequence_padded = []
    sequence_length = []

    max_length_word = max([max(map(lambda x: len(x), seq))
                           for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        # all words are same length now
        sp, sl = _pad_sequences(seq, 0, max_length_word)
        sequence_padded += [sp]
        sequence_length += [sl]

    max_length_sentence = max(map(lambda x: len(x), sequences))
    sequence_padded, _ = _pad_sequences(sequence_padded,
                                        [0] * max_length_word, max_length_sentence)
    sequence_length, _ = _pad_sequences(sequence_length, 0,
                                        max_length_sentence)
    return sequence_padded, sequence_length


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def convert_words_to_ids(location):
    words_to_ids = {}
    ids_to_words = {}
    i = 0
    for line in codecs.open(location, 'r', encoding):
        line = line.split('\n')[0].encode(encoding)

        words_to_ids[line] = i
        ids_to_words[i] = line
        i += 1

    return words_to_ids, ids_to_words


def convert_tags_to_ids(location):
    words_to_ids = {}
    ids_to_words = {}
    i = 0
    for line in codecs.open(location, 'r', encoding):
        line = line.split('\n')[0]

        words_to_ids[line] = i
        ids_to_words[i] = line
        i += 1

    return words_to_ids, ids_to_words

def build_data_test(location):

    training_data_ids = []
    training_tags_ids = []
    ids_data = []
    ids_tags = []
    words_to_ids, ids_to_words = convert_words_to_ids(cfg.vocabulary_location)
    tags_to_ids, ids_to_tags = convert_words_to_ids(
        cfg.vocabulary_tags_location)
    chars_to_ids, _ = convert_words_to_ids(cfg.vocabulary_chars_location)

    chars_in_word = []
    chars_words = []
    list_chars = []
    for sentence_words, sentence_tags in yield_data(location):
        for word, tag in zip(sentence_words, sentence_tags):

            word = word.lower()
            if words_to_ids.get(word) != None:
                ids_data.append(words_to_ids[word])

                for w in list(word.decode(encoding)):
                    if chars_to_ids.get(w.encode(encoding)) != None:
                        chars_in_word.append(chars_to_ids[w.encode(encoding)])
                chars_words.append(chars_in_word)
            else:
                f = codecs.open("./results/test.txt", 'w', encoding)
                f.write(word)
                f.write('\n')
            if tags_to_ids.get(tag) != None:
                ids_tags.append(tags_to_ids.get(tag))
        training_data_ids.append(ids_data)
        ids_data = []
        training_tags_ids.append(ids_tags)
        ids_tags = []
        list_chars.append(chars_words)

    return training_data_ids, training_tags_ids, list_chars, words_to_ids, ids_to_words


def build_data(training_data, training_tags):

    training_data_ids = []
    training_tags_ids = []
    ids_data = []
    ids_tags = []
    words_to_ids, ids_to_words = convert_words_to_ids(cfg.vocabulary_location)
    tags_to_ids, ids_to_tags = convert_tags_to_ids(
        cfg.vocabulary_tags_location)

    chars_to_ids, _ = convert_words_to_ids(cfg.vocabulary_chars_location)

    chars_in_word = []
    chars_words = []
    list_chars = []
    unk = "$UNK$".encode(encoding)
    for sentence_words, sentence_tags in zip(training_data, training_tags):

        for word, tag in zip(sentence_words, sentence_tags):

            # word = word.lower()

            # word = word.decode(encoding).replace(" ", "").encode(encoding)
            if words_to_ids.get(word.lower().encode(encoding)) != None:
                ids_data.append(words_to_ids[word.lower().encode(encoding)])
            else:

                ids_data.append(words_to_ids[unk])
            word = word.encode(encoding)
            for w in list(word.decode(encoding)):
                if chars_to_ids.get(w.encode(encoding)) != None:
                    chars_in_word.append(chars_to_ids[w.encode(encoding)])
                else:
                    chars_in_word.append(chars_to_ids[unk])

            chars_words.append(chars_in_word)
            chars_in_word = []
            if tags_to_ids.get(tag) != None:
                ids_tags.append(tags_to_ids[tag])
        training_data_ids.append(ids_data)

        ids_data = []
        training_tags_ids.append(ids_tags)
        ids_tags = []
        list_chars.append(chars_words)
        chars_words = []

    return training_data_ids, training_tags_ids, list_chars, words_to_ids, ids_to_words


def save_session(sess, saver, location):
    saver.save(sess, location)


def restore_session(location, saver, sess):
    print('Loading latest trained model')
    saver.restore(sess, location)


def predict_batch(sess, data, chars_ids_test, placeholders, parameters, batch_pos, batch_pos2, batch_pos3, batch_pos4, batch_pos5, batch_case, batch_lex, batch_gen, batch_num, batch_art):
    # sess, saver = initialize_session()
    # restore_session('./results/model', saver, sess)
    batch_ids = {
            'batch_training': data,
            'batch_chars_ids': chars_ids_test,
            'batch_pos': batch_pos,
            'batch_pos2': batch_pos2,
            'batch_pos3': batch_pos3,
            'batch_pos4': batch_pos4,
            'batch_pos5': batch_pos5,
            'batch_case': batch_case,
            'batch_lex': batch_lex,
            'batch_gen': batch_gen,
            'batch_num': batch_num,
            'batch_art': batch_art,
            'batch_tags': None
        }
    fd, sequence_lengths = get_feed(
        batch_ids, placeholders, lr=None, dropout=cfg.dropout)

    # get tag scores and transition params of CRF
    viterbi_sequences = []
    logits, trans_params = sess.run(
        [parameters['logits'], parameters['trans_params']], feed_dict=fd)

    # iterate over the sentences because no batching in vitervi_decode
    for logit, sequence_length in zip(logits, sequence_lengths):
        logit = logit[:sequence_length]  # keep only the valid steps

        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
            logit, trans_params)
        viterbi_sequences += [viterbi_seq]

    return viterbi_sequences, sequence_lengths

    
    # labels_pred = sess.run(parameters['labels_pred'], feed_dict=fd)

    # return labels_pred, sequence_lengths


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    # tags = {}
    # for i, tag in enumerate(tags_arr):
    #     # tag = tag.decode(encoding)
    #     tags[tag] = i

    default = tags['O']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def load_vocabulary_list(fif):
    tags = []
    ff = codecs.open(fif, "rb", encoding)
    for line in ff:
        tags.append(line.strip())
    return tags


def evaluate(sess, test_data_ids, test_tags_id, chars_ids_test, test_ids, placeholders, parameters):
    # batches_training_test, batches_tags_test = create_batches(test_data_ids, test_tags_id, n_test)
    f = codecs.open(cfg.result_location, 'w', encoding)
    
    accs = []
    tags = load_vocabulary_tags(cfg.vocabulary_tags_location)
    tags_ = load_vocabulary_tags_keys(cfg.vocabulary_tags_location)
    words = load_vocabulary_to_id(cfg.vocabulary_location)
    correct_preds, total_correct, total_preds = 0., 0., 0.

    correct = 0
    overall = 0
    equal = 0
    count = 0
    O, b_loc, i_loc, b_org, i_org, b_pers, i_pers, b_oth, i_oth = 0, 0, 0, 0, 0, 0, 0, 0, 0
    O_all, b_loc_all, i_loc_all, b_org_all, i_org_all, b_pers_all, i_pers_all, b_oth_all, i_oth_all = 0, 0, 0, 0, 0, 0, 0, 0, 0

    batches_training = create_batches_test(test_data_ids, cfg.batches_size)
    batches_tags = create_batches_test(test_tags_id, cfg.batches_size)
    chars_ids_test_batch = create_batches_test(chars_ids_test, cfg.batches_size)
    pos_batches = create_batches_test(test_ids['pos'], cfg.batches_size)
    pos2_batches = create_batches_test(test_ids['pos2'], cfg.batches_size)
    pos3_batches = create_batches_test(test_ids['pos3'], cfg.batches_size)
    pos4_batches = create_batches_test(test_ids['pos4'], cfg.batches_size)
    pos5_batches = create_batches_test(test_ids['pos5'], cfg.batches_size)
    case_batches = create_batches_test(test_ids['case'], cfg.batches_size)
    lex_batches = create_batches_test(test_ids['lex'], cfg.batches_size)
    gen_batches = create_batches_test(test_ids['gen'], cfg.batches_size)
    num_batches = create_batches_test(test_ids['num'], cfg.batches_size)
    art_batches = create_batches_test(test_ids['art'], cfg.batches_size)

    for i, (batch_words, batch_tags, batch_chars, batch_pos, batch_pos2, batch_pos3, batch_pos4, batch_pos5, batch_case, batch_lex, batch_gen, batch_num, batch_art) in enumerate(zip(batches_training, batches_tags, chars_ids_test_batch, pos_batches, pos2_batches, pos3_batches,  pos4_batches, pos5_batches, case_batches, lex_batches, gen_batches, num_batches, art_batches)):

        labels_pred, sequence_lengths = predict_batch(
            sess, batch_words, batch_chars, placeholders, parameters, batch_pos, batch_pos2, batch_pos3, batch_pos4, batch_pos5, batch_case, batch_lex, batch_gen, batch_num, batch_art)

        for lab, lab_pred, length, sentence in zip(batch_tags, labels_pred,
                                                   sequence_lengths, batch_words):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            accs += [a == b for (a, b) in zip(lab, lab_pred)]
            for a, b, s in zip(lab, lab_pred, sentence):
                f.write(words.get(s).decode(encoding))
                f.write(' ')
                f.write(tags_[a])
                f.write(' ')
                f.write(tags_[b])
                f.write("\n")
            f.write("\n")
            lab_chunks = set(get_chunks(lab, tags))

            lab_pred_chunks = set(get_chunks(lab_pred, tags))

            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)
    # print("correct predictions", correct_preds)
    # print("total predictions", total_preds)
    # print("total correct", total_correct)
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    acc = np.mean(accs)
    # print("precision", p)
    # print("recall", r)

    print("acc", 100 * acc, "f1", f1 * 100)
    return f1 * 100


def run_evaluate(location):
    print('a')
    # labels_pred, sequence_lengths = predict_batch(
    #         sess, batch_words, batch_chars, placeholders, parameters)

    # for lab, lab_pred, length, sentence in zip(batch_tags, labels_pred,
    #                                             sequence_lengths, batch_words):

    #     lab = lab[:length]
    #     lab_pred = lab_pred[:length]

    # training_data_ids, training_tags_ids, words_to_ids, ids_to_words = build_data(
    #     location)
    # test_data_ids, test_tags_ids, _, _ = build_data(test_location)
    # placeholders = define_placeholders()
    # parameters = build_model(training_data_ids, words_to_ids, placeholders)
    # sess, saver = initialize_session()
    # n = int(len(training_data_ids) / num_batches)
    # n_test = int(len(test_data_ids) / num_batches)

    # batches_training, batches_tags = create_batches(
    #     training_data_ids, training_tags_ids, n)

    # evaluate(test_data_ids, test_tags_id, placeholders, parameters)


def main():
    # randomize_data()
    train_model()
    # run_evaluate('./dataset/bg/testb.txt')
    # get_fasttext_vocab(pretrained_vectors_location)
    # evaluate('./dataset/bg/testa.txt')


if __name__ == '__main__':
    main()

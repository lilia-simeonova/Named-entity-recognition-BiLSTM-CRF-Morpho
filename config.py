encoding = 'utf-8'

vocabulary_location = './vocabulary/words-bg-new'
vocabulary_tags_location = './vocabulary/tags-bg-new'
vocabulary_chars_location = './vocabulary/chars-bg-new'
model = './results/models/new_res-new'

# vocabulary_location = './vocabulary/words'
# vocabulary_tags_location = './vocabulary/tags'
# vocabulary_chars_location = './vocabulary/chars'
# model = './results/model-en'


all_data = './dataset/bg/all_data'

train_location = './dataset/new_train.txt'
test_location = './dataset/new_test.txt'
dev_location = './dataset/new_test.txt'

# train_location = './dataset-en/train.txt'
# test_location = './dataset-en/testb.txt'
# dev_location = './dataset-en/testb.txt'

result_location = "./results/tagged/new_res-bg-test-all"

pretrained_vectors_location = './embeddings/wiki.bg.vec'
trimmed_embeddings_file = 'embeddings/trimmed.npz'

# pretrained_vectors_location = './embeddings/glove.6B.300d.txt'
# trimmed_embeddings_file = 'embeddings/trimmed-en.npz'

# pretrained_vectors_location = './embeddings/glove.6B.300d.txt'
# trimmed_embeddings_file = 'embeddings/en-trimmed.npz'

_lr_m = 'adam'
dropout = 2
lr = 0.001
lr_decay = 0.9
hidden_size_lstm = 300
hidden_size_char = 100
dim = 300
dim_char = 100
ntags = 9
clip = 1

use_en = False

use_pos = True
use_pos2 = False
use_pos3 = True
use_pos4 = False
use_pos5 = False

use_morf = True
use_gen = True
use_art = True
use_num = True

use_chars = True
use_words = True

use_case = False

use_lex = True

replace = False
# may want to change the emb size of chars to 100 in both places

batches_size = 20
nepoch_no_imprv = 4

epoch_range = 100
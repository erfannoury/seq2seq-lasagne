import numpy as np
from os import path
from nltk.tokenize import TreebankWordTokenizer
from itertools import *
import cPickle as pickle

BASE_PATH = 'E:/University Central/Bachelor Thesis/Seq2Seq/seq2seq-lasagne/data/cornell movie-dialogs corpus/'

# copied from the Python documentation
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

class CornellMovieDialogs:
    """
    
    """
    def __init__(self, base_path):
        """
        Path to where these two files of the Cornell Movie Dialogs corpus is present:
            movie_conversations.txt
            movie_lines.txt
        """
        self.BASE_PATH = base_path
        self.CONVS_FILE = 'movie_conversations.txt'
        self.LINES_FILE = 'movie_lines.txt'
        self.FILE_SEP = '+++$+++'
        self.EOS_TOKEN = '<EOS>'
        self.GO_TOKEN = '<GO>'
        self.PAD_TOKEN = '<PAD>'
        self.MAX_ENC_MB_SIZE = 4096
        self.MAX_DEC_MB_SIZE = 2048
        self.VAL_COUNT = 600
        self.bucket_sizes = [(15, 31), (15, 63), (31, 31), (63, 63), (63, 127), 
                            (127, 127), (31, 255), (255, 63), (255, 127), (63, 255), 
                            (31, 511)]
        self.bucket_minibatch_sizes = {}
        self.bucket_pairs = {}
        for size in self.bucket_sizes:
            self.bucket_pairs[size] = []
            self.bucket_minibatch_sizes[size] = int(np.min((np.ceil(self.MAX_ENC_MB_SIZE / (size[0] + 1)), 
                                                            np.ceil(self.MAX_DEC_MB_SIZE / (size[1] + 1) ))))
        self.bucket_pairs[(-1, -1)] = []
        self.bucket_minibatch_sizes[(-1, -1)] = 2
        self.__prepare__()

    def __prepare__(self):
        """
        
        """
        conversations = open(path.join(self.BASE_PATH, self.CONVS_FILE), 'r').readlines()
        movie_lines = open(path.join(self.BASE_PATH, self.LINES_FILE), 'r').readlines()
        tbt = TreebankWordTokenizer().tokenize
        self.words_set = set()
        self.lines_dict = {}
        for i, line in enumerate(movie_lines):
            parts = map(lambda x: x.strip(), line.lower().split(self.FILE_SEP))
            tokens = tbt(parts[-1])
            self.lines_dict[parts[0]] = tokens
            self.words_set |= set(tokens)
        self.word2idx = {}
        self.word2idx[self.PAD_TOKEN] = 0
        self.word2idx[self.EOS_TOKEN] = 1
        self.word2idx[self.GO_TOKEN] = 2
        for i, word in enumerate(self.words_set):
            self.word2idx[word] = i + 3
        self.idx2word = [0] * len(self.word2idx)
        for w, i in self.word2idx.items():
            self.idx2word[i] = w

        # extract pairs of lines in a conversation (s0, s1, s2) -> {(s0, s1), (s1, s2)}
        utt_pairs = []
        for line in conversations:
            parts = map(lambda x: x[1:-1], map(lambda x: x.strip(), line.lower().split(self.FILE_SEP))[-1][1:-1].split(', '))
            utt_pairs += list(pairwise(parts))
        utt_pairs = np.random.permutation(utt_pairs)
        train_utt_pairs = utt_pairs[self.VAL_COUNT:]
        self.val_pairs = utt_pairs[:self.VAL_COUNT]

        def find_bucket(enc_size, dec_size, buckets):
            return next(dropwhile(lambda x: enc_size > x[0] or dec_size > x[1], buckets), None)

        for pair in train_utt_pairs:
            bckt = find_bucket(len(self.lines_dict[pair[0]]), len(self.lines_dict[pair[1]]), self.bucket_sizes)
            if bckt is None:
                self.bucket_pairs[(-1, -1)].append(pair)
            else:
                self.bucket_pairs[bckt].append(pair)

        self.bucket_ordering = []
        for bckt, _ in sorted(map(lambda x: (x[0], len(x[1])), self.bucket_pairs.items()), key=lambda x: x[1], reverse=True):
            self.bucket_ordering.append(bckt)


    def __grouper__(self, iterable, n, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        args = [iter(np.random.permutation(iterable))] * n
        return izip_longest(fillvalue=fillvalue, *args)


    def __create_row__(self, tokens, max_len, kind):
        """
        kind:
            enc_in  -> reversed sequence, <EOS>, <PAD>
            dec_in  -> <GO>, the sequence, <PAD>
            dec_out -> the sequence, <EOS>, <PAD>
        """
        row = np.zeros((max_len, ), dtype=np.int32)
        if kind == 'enc_in':
            tokens = tokens[::-1]
            for i, t in enumerate(tokens):
                row[i] = self.word2idx[t]
            row[len(tokens)] = self.word2idx[self.EOS_TOKEN]
        elif kind == 'dec_in':
            row[0] = self.word2idx[self.GO_TOKEN]
            for i, t in enumerate(tokens):
                row[i+1] = self.word2idx[t]
        elif kind == 'dec_out':
            for i, t in enumerate(tokens):
                row[i] = self.word2idx[t]
            row[len(tokens)] = self.word2idx[self.EOS_TOKEN]
        else:
            raise ValueError('kind must be one of {enc_in, dec_in, dec_out}')
        return row


    # TODO: Multi-threaded iterator
    def train_iterator(self):
        """
        TODO: Change the iterator to be multi-threaded
        Returns
        =======
        encoder_input: np.ndarray
            input to the encoder
        encoder_mask: np.ndarray
            mask for the input to the encoder
        decoder_input: np.ndarray
            input to the decoder
        decoder_mask : np.ndarray
            mask for the input to the decoder
        decoder_output : np.ndarray
            expected output of the decoder
        """
        for bucket in self.bucket_ordering:
            for pair_group in self.__grouper__(self.bucket_pairs[bucket], self.bucket_minibatch_sizes[bucket]):
                pair_group = filter(lambda x: x is not None, pair_group)
                if bucket == (-1, -1): # the long pairs!
                    max_len_enc = max(map(lambda x: len(self.lines_dict[x[0]]), pair_group)) + 1
                    max_len_dec = max(map(lambda x: len(self.lines_dict[x[1]]), pair_group)) + 1
                else:
                    max_len_enc = bucket[0] + 1
                    max_len_dec = bucket[1] + 1
                encoder_input = np.zeros((len(pair_group), max_len_enc), dtype=np.int32)
                decoder_input = np.zeros((len(pair_group), max_len_dec), dtype=np.int32)
                decoder_output = np.zeros((len(pair_group), max_len_dec), dtype=np.int32)
                for i, pair in enumerate(pair_group):
                    encoder_input[i, :] = self.__create_row__(self.lines_dict[pair[0]], max_len_enc, 'enc_in')
                    decoder_input[i, :] = self.__create_row__(self.lines_dict[pair[1]], max_len_dec, 'dec_in')
                    decoder_output[i, :]= self.__create_row__(self.lines_dict[pair[1]], max_len_dec, 'dec_out')
                yield encoder_input, (encoder_input > 0), decoder_input, (decoder_input > 0), decoder_output


if __name__ == '__main__':
    cornell = CornellMovieDialogs(BASE_PATH)
    for ei, em, di, dm, do in cornell.train_iterator():
        print ei.shape, em.shape, di.shape, dm.shape, do.shape
        print ei
        print '#########################'
        print di
        print '#########################'
        print do
        break

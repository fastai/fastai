"NLP data processing; tokenizes text and creates vocab indexes"
from ..torch_core import *

__all__ = ['BaseTokenizer', 'SpacyTokenizer', 'Tokenizer', 'Vocab', 'deal_caps', 'fix_html', 'replace_rep', 'replace_wrep', 
           'rm_useless_spaces', 'spec_add_spaces', 'BOS', 'FLD', 'UNK', 'PAD', 'TK_UP', 'TK_REP', 'TK_REP', 'TK_WREP', 
           'default_rules', 'default_spec_tok']

BOS,FLD,UNK,PAD = 'xxbos','xxfld','xxunk','xxpad'
TK_UP,TK_REP,TK_WREP = 'xxup','xxrep','xxwrep'


class BaseTokenizer():
    "Basic class for a tokenizer function."
    def __init__(self, lang:str):                      self.lang = lang
    def tokenizer(self, t:str) -> List[str]:           return t.split(' ')
    def add_special_cases(self, toks:Collection[str]): pass

class SpacyTokenizer(BaseTokenizer):
    "Wrapper around a spacy tokenizer to make it a `BaseTokenizer`."

    def __init__(self, lang:str):
        self.tok = spacy.blank(lang)

    def tokenizer(self, t:str) -> List[str]:
        return [t.text for t in self.tok.tokenizer(t)]

    def add_special_cases(self, toks:Collection[str]):
        for w in toks:
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])

def spec_add_spaces(t:str) -> str:
    "Add spaces around / and # in `t`."
    return re.sub(r'([/#])', r' \1 ', t)

def rm_useless_spaces(t:str) -> str:
    "Remove multiple spaces in `t`."
    return re.sub(' {2,}', ' ', t)

def replace_rep(t:str) -> str:
    "Replace repetitions at the character level in `t`."
    def _replace_rep(m:Collection[str]) -> str:
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '
    re_rep = re.compile(r'(\S)(\1{3,})')
    return re_rep.sub(_replace_rep, t)

def replace_wrep(t:str) -> str:
    "Replace word repetitions in `t`."
    def _replace_wrep(m:Collection[str]) -> str:
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '
    re_wrep = re.compile(r'(\b\w+\W+)(\1{3,})')
    return re_wrep.sub(_replace_wrep, t)

def deal_caps(t:str) -> str:
    "Replace words in all caps in `t`."
    res = []
    for s in re.findall(r'\w+|\W+', t):
        res += ([f' {TK_UP} ',s.lower()] if (s.isupper() and (len(s)>2)) else [s.lower()])
    return ''.join(res)

def fix_html(x:str) -> str:
    "List of replacements from html strings in `x`."
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>',UNK).replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

default_rules = [fix_html, replace_rep, replace_wrep, deal_caps, spec_add_spaces, rm_useless_spaces]
default_spec_tok = [BOS, FLD, UNK, PAD]

class Tokenizer():
    "Put together rules, a tokenizer function and a language to tokenize text with multiprocessing."
    def __init__(self, tok_func:Callable=SpacyTokenizer, lang:str='en', rules:ListRules=None,
                 special_cases:Collection[str]=None, n_cpus:int=None):
        self.tok_func,self.lang,self.special_cases = tok_func,lang,special_cases
        self.rules = rules if rules else default_rules
        self.special_cases = special_cases if special_cases else default_spec_tok
        self.n_cpus = n_cpus or num_cpus()//2

    def __repr__(self) -> str:
        res = f'Tokenizer {self.tok_func.__name__} in {self.lang} with the following rules:\n'
        for rule in self.rules: res += f' - {rule.__name__}\n'
        return res

    def process_text(self, t:str, tok:BaseTokenizer) -> List[str]:
        "Processe one text `t` with tokenizer `tok`."
        for rule in self.rules: t = rule(t)
        return tok.tokenizer(t)

    def _process_all_1(self, texts:Collection[str]) -> List[List[str]]:
        "Process a list of `texts` in one process."
        tok = self.tok_func(self.lang)
        if self.special_cases: tok.add_special_cases(self.special_cases)
        return [self.process_text(t, tok) for t in texts]

    def process_all(self, texts:Collection[str]) -> List[List[str]]:
        "Process a list of `texts`."
        if self.n_cpus <= 1: return self._process_all_1(texts)
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(e.map(self._process_all_1, partition_by_cores(texts, self.n_cpus)), [])

class Vocab():
    "Contain the correspondance between numbers and tokens and numericalize."

    def __init__(self, path:PathOrStr):
        self.itos = pickle.load(open(Path(path)/'itos.pkl', 'rb'))
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return [self.stoi[w] for w in t]

    def textify(self, nums:Collection[int]) -> List[str]:
        "Convert a list of `nums` to their tokens."
        return ' '.join([self.itos[i] for i in nums])

    @classmethod
    def create(cls, path:PathOrStr, tokens:Tokens, max_vocab:int, min_freq:int) -> 'Vocab':
        "Create a vocabulary from a set of tokens."
        freq = Counter(p for o in tokens for p in o)
        itos = [o for o,c in freq.most_common(max_vocab) if c > min_freq]
        itos.insert(0, PAD)
        if UNK in itos: itos.remove(UNK)
        itos.insert(0, UNK)
        pickle.dump(itos, open(Path(path)/'itos.pkl', 'wb'))
        h = hashlib.sha1(np.array(itos))
        with open(Path(path)/'numericalize.log','w') as f: f.write(h.hexdigest())
        return cls(path)

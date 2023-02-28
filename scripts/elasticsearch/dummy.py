import random

words = ['musical', 'soundtrack', 'conspiracy', 'mercy', 'composed', 'speak', 'elder', 'pool', 'combination', 'frequency', 'pressure', 'wars', 'crime', 'deny', 'captain', 'roll', 'though', 'regulation', 'involved', 'paul', 'rates', 'lemon', 'dover', 'association', 'blog', 'converted', 'checklist', 'green', 'powder', 'acres', 'repository', 'crash', 'organisms', 'card', 'side', 'impressive', 'achieve', 'scott', 'subsequently', 'mozambique', 'brand', 'sucking', 'jerusalem', 'doctrine', 'acute', 'continental', 'cleaner', 'aggressive', 'hydrocodone', 'childhood', 'census', 'schemes', 'proof', 'mystery', 'back', 'stands', 'arch', 'told', 'dosage', 'gardening', 'compiled', 'object', 'elliott', 'monitored', 'terminology', 'says', 'compilation', 'jesus', 'dates', 'islands', 'ball', 'british', 'playback', 'oxygen', 'porsche', 'stick', 'stopped', 'cube', 'further', 'exists', 'evolution', 'organized', 'hormone', 'soldiers', 'general', 'anthony', 'juvenile', 'saves', 'divx', 'trouble', 'sage', 'merit', 'necessary', 'catalyst', 'abstract', 'andrews', 'butt', 'constraint', 'flesh', 'judges', 'fundamentals', 'reprint', 'colin', 'sunday', 'another', 'chemical', 'scripting', 'uploaded', 'korea', 'entered', 'promotions', 'partner', 'knew', 'heath', 'drill', 'related', 'fitted', 'globe', 'sacramento', 'diagnostic', 'consideration', 'abraham', 'temperature', 'young', 'premises', 'tunnel', 'teeth', 'stability', 'posted', 'films', 'symposium', 'seeing', 'rely', 'substitute', 'installing', 'scotland', 'semester', 'think', 'experiencing', 'infant', 'rocket', 'couple', 'gentle', 'gary', 'plug', 'minimal', 'capabilities', 'mainland', 'employers', 'berry', 'adequate', 'scottish', 'kenya', 'palm', 'homeless', 'talked', 'upgrade', 'firms', 'story', 'helmet', 'sword', 'sand', 'humans', 'tourist', 'melbourne', 'harley', 'according', 'entering', 'frankfurt', 'authority', 'reviewer', 'promising', 'afraid', 'ivory', 'agriculture', 'readily', 'drawings', 'confirm', 'exotic', 'buried', 'shower', 'intimate', 'testimony', 'severe', 'commonly', 'relay', 'counsel', 'demonstrates', 'obesity', 'cigarette', 'about', 'maternity', 'quote', 'seminar', 'profiles', 'gordon', 'tubes', 'stays', 'begin', 'heroes', 'wisdom', 'wagner', 'tribes', 'employ', 'platinum', 'practice', 'bonds', 'preparation', 'communities', 'shopzilla', 'station', 'contacted', 'jamie', 'stereo', 'current', 'errors', 'endorsement', 'optimum', 'downtown', 'towards', 'comparisons', 'reserved', 'fault', 'instructors', 'duration', 'rose', 'deaths', 'probability', 'bits', 'officers', 'acrylic', 'some', 'however', 'participating', 'paypal', 'excluding', 'signature', 'enemy', 'wait', 'promotional', 'pill', 'military', 'stopping', 'premiere', 'webcast', 'silly', 'oral', 'drain', 'necklace', 'inventory', 'tomato', 'gaming', 'jewel', 'startup', 'doctors', 'sterling', 'manually', 'workforce', 'anyway', 'evil', 'bargain', 'equally', 'biodiversity', 'remedies', 'facilities', 'dylan', 'nigeria', 'women', 'steam', 'athletic', 'balanced', 'arms', 'destiny', 'guardian', 'paper', 'investing', 'introduce', 'mime', 'sends', 'lovely', 'appreciated', 'restoration', 'roller', 'magical', 'waiting', 'tender', 'removing', 'defence', 'notebooks', 'drew', 'vampire', 'columns', 'unique', 'mount', 'representative', 'diesel', 'progress', 'procedure', 'heritage', 'bloomberg', 'vanilla', 'diamond', 'lexington', 'spent', 'acrobat', 'metropolitan', 'aerial', 'auctions', 'senators', 'layers', 'replace', 'signal', 'plant', 'prep', 'assisted', 'mods', 'newark', 'heavily', 'drugs', 'javascript', 'multimedia', 'dealtime', 'graduates', 'championship', 'basics', 'limits', 'parties', 'requests', 'florist', 'preferred', 'those', 'contacting', 'asian', 'owned', 'beginning', 'format', 'likewise', 'encouraging', 'composite', 'chemistry', 'varies', 'currencies', 'movie', 'champions', 'concluded', 'promoting', 'liable', 'symptoms', 'guru', 'assignments', 'dynamics', 'prisoner', 'garmin', 'dude', 'independent', 'thesis', 'offensive', 'itself', 'yale', 'action', 'requesting', 'blake', 'mario', 'photographers', 'happening', 'capable']


def gen_random_phrase(min_words, max_words):
    n_words = random.randint(min_words, max_words)
    return ' '.join(random.sample(words, n_words)).capitalize()


def gen_random_docs(n):
    docs = []
    for i in range(n):
        text = gen_random_phrase(10, 100)
        opening_text = text[:(len(text) // 2)]

        doc = {
            'id': i,
            'title': gen_random_phrase(2, 4),
            'category': [gen_random_phrase(1, 2) for j in range(random.randint(1, 2))],
            'heading': [gen_random_phrase(1, 2) for j in range(random.randint(1, 2))],
            'redirect': [gen_random_phrase(1, 2) for j in range(random.randint(1, 2))],
            'text': text,
            'opening_text': opening_text,
            'auxiliary_text': gen_random_phrase(1, 6),
            'file_text': gen_random_phrase(0, 1),
            'popularity_score': random.random(),
            'incoming_links': random.randint(1, 10000)
        }
        docs.append(doc)
    return docs

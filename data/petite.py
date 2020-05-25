"""

	Dataset loading en-ro en-jp dataset

"""
from data.annotated import AnnotatedTextDataset


class PETITEEnRoDataset(AnnotatedTextDataset):
    NAME = "wmt"
    LANGUAGE_PAIR = ('en', 'ro')

    URLS = [
        ('europarl.tgz', 'http://data.statmt.org/wmt16/translation-task/training-parallel-ep-v8.tgz'),
        ('setimes2.zip', 'http://opus.nlpl.eu/download.php?f=SETIMES/v2/moses/en-ro.txt.zip'),
        ('dev.tgz', 'http://data.statmt.org/wmt16/translation-task/dev.tgz'),
        ('test.tgz', 'http://data.statmt.org/wmt16/translation-task/test.tgz')
    ]

    RAW_SPLITS = {
        'train': [
            ('training-parallel-ep-v8/europarl-v8.ro-en.en', 'training-parallel-ep-v8/europarl-v8.ro-en.ro'),
            ('en-ro/SETIMES.en-ro.en', 'en-ro/SETIMES.en-ro.ro')
        ],
        'dev': [
            ('dev/newsdev2016-roen-ref.en.sgm', 'dev/newsdev2016-enro-ref.ro.sgm')
        ],
        'test': [
            ('test/newstest2016-roen-ref.en.sgm', 'test/newstest2016-enro-ref.ro.sgm')
        ]
    }

    SPLITS = {
        'train': 'train.tok',
        'dev': 'dev.tok',
        'test': 'test.tok'
    }


class PETITEEnJpDataset(AnnotatedTextDataset):
    NAME = "iwslt"
    LANGUAGE_PAIR = ('en', 'ja')
    WORD_COUNT = (5114050, 3576290)

    URLS = [
        ('iwslt_en_ja.tgz', 'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/ja/en-ja.tgz'),
        ('iwslt_en_ja_test.tgz',
         'https://wit3.fbk.eu/download.php?release=2017-01-ted-test&type=texts&slang=en&tlang=ja')
    ]

    RAW_SPLITS = {
        'train': [
            ('en-ja/train.tags.en-ja.en', 'en-ja/train.tags.en-ja.ja')
        ],
        'dev': [
            ('en-ja/IWSLT17.TED.dev2010.en-ja.en.xml', 'en-ja/IWSLT17.TED.dev2010.en-ja.ja.xml'),
            ('en-ja/IWSLT17.TED.tst2010.en-ja.en.xml', 'en-ja/IWSLT17.TED.tst2010.en-ja.ja.xml'),
            ('en-ja/IWSLT17.TED.tst2011.en-ja.en.xml', 'en-ja/IWSLT17.TED.tst2011.en-ja.ja.xml'),
            ('en-ja/IWSLT17.TED.tst2012.en-ja.en.xml', 'en-ja/IWSLT17.TED.tst2012.en-ja.ja.xml'),
            ('en-ja/IWSLT17.TED.tst2013.en-ja.en.xml', 'en-ja/IWSLT17.TED.tst2013.en-ja.ja.xml'),
            ('en-ja/IWSLT17.TED.tst2014.en-ja.en.xml', 'en-ja/IWSLT17.TED.tst2014.en-ja.ja.xml'),
            ('en-ja/IWSLT17.TED.tst2015.en-ja.en.xml', 'en-ja/IWSLT17.TED.tst2015.en-ja.ja.xml')
        ],
        'test': [
            ('en-de/IWSLT17.TED.tst2016.en-ja.en.xml', 'ja-en/IWSLT17.TED.tst2016.ja-en.ja.xml'),
            ('en-de/IWSLT17.TED.tst2017.en-ja.en.xml', 'ja-en/IWSLT17.TED.tst2017.ja-en.ja.xml')
        ]
    }
    SPLITS = {
        'train': 'train.tok',
        'dev': 'dev.tok',
        'test': 'test.tok'
    }

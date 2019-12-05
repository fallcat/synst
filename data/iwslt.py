'''
Data loading and pre-processing for the IWSLT'16 EN-DE dataset.
'''
import re

from data.annotated import AnnotatedTextDataset


class IWSLTDataset(AnnotatedTextDataset):
    ''' Class that encapsulates the IWSLT dataset '''
    NAME = 'iwslt'
    LANGUAGE_PAIR = ('en', 'de')
    # WORD_COUNT = (4215814, 4186988)
    WORD_COUNT = (1.0360595565014956, 1)

    URLS = [
        ('iwslt_en_de.tgz', 'https://wit3.fbk.eu/archive/2016-01/texts/en/de/en-de.tgz'),
        ('iwslt_test_en_de.tgz', 'https://wit3.fbk.eu/archive/2016-01-test/texts/en/de/en-de.tgz'),
        ('iwslt_test_de_en.tgz', 'https://wit3.fbk.eu/archive/2016-01-test/texts/de/en/de-en.tgz'),
    ]
    RAW_SPLITS = {
        'train': [
            ('en-de/train.tags.en-de.en', 'en-de/train.tags.en-de.de')
        ],
        'dev': [
            ('en-de/IWSLT16.TED.tst2013.en-de.en.xml', 'en-de/IWSLT16.TED.tst2013.en-de.de.xml'),
        ],
        'valid': [
            ('en-de/IWSLT16.TED.dev2010.en-de.en.xml', 'en-de/IWSLT16.TED.dev2010.en-de.de.xml'),
            ('en-de/IWSLT16.TED.tst2010.en-de.en.xml', 'en-de/IWSLT16.TED.tst2010.en-de.de.xml'),
            ('en-de/IWSLT16.TED.tst2011.en-de.en.xml', 'en-de/IWSLT16.TED.tst2011.en-de.de.xml'),
            ('en-de/IWSLT16.TED.tst2012.en-de.en.xml', 'en-de/IWSLT16.TED.tst2012.en-de.de.xml'),
            ('en-de/IWSLT16.TED.tst2013.en-de.en.xml', 'en-de/IWSLT16.TED.tst2013.en-de.de.xml'),
            ('en-de/IWSLT16.TED.tst2014.en-de.en.xml', 'en-de/IWSLT16.TED.tst2014.en-de.de.xml'),
        ],
        'test': [
            ('en-de/IWSLT16.QED.tst2016.en-de.en.xml', 'de-en/IWSLT16.QED.tst2016.de-en.de.xml'),
            ('en-de/IWSLT16.TED.tst2015.en-de.en.xml', 'de-en/IWSLT16.TED.tst2015.de-en.de.xml'),
            ('en-de/IWSLT16.TED.tst2016.en-de.en.xml', 'de-en/IWSLT16.TED.tst2016.de-en.de.xml'),
        ]
    }
    SPLITS = {
        'train': 'train.tok',
        'valid': 'valid.tok',
        'dev': 'dev.tok',
        'test': 'test.tok'
    }

    IGNORE_REGEX_LIST = [
        re.compile(fr'<\s*{tag}\s*[^>]*\s*>[^<]*<\s*/{tag}\s*>')
        for tag in
        (
            'url', 'keywords', 'speaker', 'talkid',
            'title', 'description', 'reviewer', 'translator'
        )
    ]


class IWSLTEnViDataset(AnnotatedTextDataset):
    ''' Class that encapsulates the IWSLT dataset '''
    NAME = 'iwslt'
    LANGUAGE_PAIR = ('en', 'vi')
    # WORD_COUNT = (4215814, 4186988)
    WORD_COUNT = (1.0360595565014956, 1)

    URLS = [
        ('iwslt_en_vi.tgz', 'https://wit3.fbk.eu/archive/2015-01/texts/en/vi/en-vi.tgz'),
        ('iwslt_test_en_vi.tgz', 'https://wit3.fbk.eu/archive/2015-01-test/texts/en/vi/en-vi.tgz'),
        ('iwslt_test_vi_en.tgz', 'https://wit3.fbk.eu/archive/2015-01-test/texts/vi/en/vi-en.tgz'),
    ]
    RAW_SPLITS = {
        'train': [
            ('en-de/train.tags.en-de.en', 'en-de/train.tags.en-de.de')
        ],
        'dev': [
            ('en-de/IWSLT15.TED.tst2013.en-vi.en.xml', 'en-de/IWSLT15.TED.tst2013.en-vi.de.xml'),
        ],
        'valid': [
            ('en-vi/IWSLT15.TED.dev2010.en-vi.en.xml', 'en-vi/IWSLT15.TED.dev2010.en-vi.vi.xml'),
            ('en-vi/IWSLT15.TED.tst2010.en-vi.en.xml', 'en-vi/IWSLT15.TED.tst2010.en-vi.vi.xml'),
            ('en-vi/IWSLT15.TED.tst2011.en-vi.en.xml', 'en-vi/IWSLT15.TED.tst2011.en-vi.vi.xml'),
            ('en-vi/IWSLT15.TED.tst2012.en-vi.en.xml', 'en-vi/IWSLT15.TED.tst2012.en-vi.vi.xml'),
            ('en-vi/IWSLT15.TED.tst2013.en-vi.en.xml', 'en-vi/IWSLT15.TED.tst2013.en-vi.vi.xml'),
        ],
        'test': [
            ('en-vi/IWSLT16.QED.tst2016.en-vi.en.xml', 'de-vi/IWSLT16.QED.tst2016.vi-en.vi.xml'),
            ('en-vi/IWSLT16.TED.tst2015.en-vi.en.xml', 'de-vi/IWSLT16.TED.tst2015.vi-en.vi.xml'),
            ('en-vi/IWSLT16.TED.tst2016.en-vi.en.xml', 'de-vi/IWSLT16.TED.tst2016.vi-en.vi.xml'),
        ]
    }
    SPLITS = {
        'train': 'train.tok',
        'valid': 'valid.tok',
        'dev': 'dev.tok',
        'test': 'test.tok'
    }

    IGNORE_REGEX_LIST = [
        re.compile(fr'<\s*{tag}\s*[^>]*\s*>[^<]*<\s*/{tag}\s*>')
        for tag in
        (
            'url', 'keywords', 'speaker', 'talkid',
            'title', 'description', 'reviewer', 'translator'
        )
    ]


class IWSLTEnJaDataset(AnnotatedTextDataset):
    ''' Class that encapsulates the IWSLT dataset '''
    NAME = 'iwslt'
    LANGUAGE_PAIR = ('en', 'ja')
    # WORD_COUNT = (4215814, 4186988)
    WORD_COUNT = (1.0360595565014956, 1)

    URLS = [
        ('iwslt_en_de.tgz', 'https://wit3.fbk.eu/archive/2016-01/texts/en/de/en-de.tgz'),
        ('iwslt_test_en_de.tgz', 'https://wit3.fbk.eu/archive/2016-01-test/texts/en/de/en-de.tgz'),
        ('iwslt_test_de_en.tgz', 'https://wit3.fbk.eu/archive/2016-01-test/texts/de/en/de-en.tgz'),
    ]
    RAW_SPLITS = {
        'train': [
            ('en-de/train.tags.en-de.en', 'en-de/train.tags.en-de.de')
        ],
        'dev': [
            ('en-de/IWSLT16.TED.tst2013.en-de.en.xml', 'en-de/IWSLT16.TED.tst2013.en-de.de.xml'),
        ],
        'valid': [
            ('en-de/IWSLT16.TED.dev2010.en-de.en.xml', 'en-de/IWSLT16.TED.dev2010.en-de.de.xml'),
            ('en-de/IWSLT16.TED.tst2010.en-de.en.xml', 'en-de/IWSLT16.TED.tst2010.en-de.de.xml'),
            ('en-de/IWSLT16.TED.tst2011.en-de.en.xml', 'en-de/IWSLT16.TED.tst2011.en-de.de.xml'),
            ('en-de/IWSLT16.TED.tst2012.en-de.en.xml', 'en-de/IWSLT16.TED.tst2012.en-de.de.xml'),
            ('en-de/IWSLT16.TED.tst2013.en-de.en.xml', 'en-de/IWSLT16.TED.tst2013.en-de.de.xml'),
            ('en-de/IWSLT16.TED.tst2014.en-de.en.xml', 'en-de/IWSLT16.TED.tst2014.en-de.de.xml'),
        ],
        'test': [
            ('en-de/IWSLT16.QED.tst2016.en-de.en.xml', 'de-en/IWSLT16.QED.tst2016.de-en.de.xml'),
            ('en-de/IWSLT16.TED.tst2015.en-de.en.xml', 'de-en/IWSLT16.TED.tst2015.de-en.de.xml'),
            ('en-de/IWSLT16.TED.tst2016.en-de.en.xml', 'de-en/IWSLT16.TED.tst2016.de-en.de.xml'),
        ]
    }
    SPLITS = {
        'train': 'train.tok',
        'valid': 'valid.tok',
        'dev': 'dev.tok',
        'test': 'test.tok'
    }

    IGNORE_REGEX_LIST = [
        re.compile(fr'<\s*{tag}\s*[^>]*\s*>[^<]*<\s*/{tag}\s*>')
        for tag in
        (
            'url', 'keywords', 'speaker', 'talkid',
            'title', 'description', 'reviewer', 'translator'
        )
    ]

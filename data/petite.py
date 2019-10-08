"""
	
	Dataset loading iwslt-sized en-fr en-ro en-jp dataset

"""

import copy

from data.annotated import AnnotatedTextDataset

class PETITEEnFrDataset(AnnotatedTextDataset):
	NAME="petite"
	LANGUAGE_PAIR = ('en', 'fr')


class PETITEEnRoDataset(AnnotatedTextDataset):
	NAME="petite"
	LANGUAGE_PAIR = ('en', 'ro')


class PETITEEnJpDataset(AnnotatedTextDataset):
	NAME="petite"
	LANGUAGE_PAIR = ('en', 'jp')
	

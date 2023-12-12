
from keywords_suggester.bin.modules.BertSingle import BertTopicClass
from keywords_suggester.config import settings


BERT = BertTopicClass(restore=1)

BERT.GenereateTopicLabels()
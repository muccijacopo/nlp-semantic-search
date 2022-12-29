import re
from typing import Dict

ATTRIBUTES = ['Id', 'PostTypeId', 'ParentId', 'CreationDate', 'Score', 'ViewCount', 'FavoriteCount', 'Title', 'Body', 'Tags', 'Topic']


def normalize_string_alt(s: str) -> str:
    return re.sub('(<.{1,5}>)|\n', "", s)


def normalize_string(s: str) -> str:
    return re.sub('\n', "", s)


def get_post_attribute(post: Dict[str, str], attr: str, topic: str) -> str:
    if attr != 'Topic':
        return post.get(attr) or ""
    else:
        return topic


def get_post_attributes(post: Dict[str, str], topic: str) -> Dict[str, str]:
    r = {}
    for a in ATTRIBUTES:
        r[a] = get_post_attribute(post, a, topic)
    return r

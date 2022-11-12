import re
from typing import Dict

ATTRIBUTES = ['Id', 'PostTypeId', 'CreationDate', 'Score', 'ViewCount', 'FavoriteCount', 'Title', 'Body', 'Tags']

# def normalize_string(s: str) -> str:
#     return re.sub('(<.{1,5}>)|\n', "", s)


def normalize_string(s: str) -> str:
    return re.sub('\n', "", s)


def get_post_attrib(post: Dict[str, str]) -> Dict[str, str]:
    r = {}
    for a in ATTRIBUTES:
        r[a] = post.get(a) or ""
    return r

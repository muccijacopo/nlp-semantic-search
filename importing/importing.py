import os
import xml.etree.ElementTree as ET
import csv
from typing import List, Dict

from utils import get_post_attributes, ATTRIBUTES, normalize_string


def get_abs_path(path: str):
    return os.path.abspath(os.path.dirname(__file__)) + path


def import_xml_to_csv():
    for name in os.listdir('./data/stackexchange'):
        dir_path = os.path.join('./data/stackexchange', name)
        for file in os.listdir(dir_path):
            if file == 'Posts.xml':
                print(os.path.join(dir_path, file))
                read_xml(os.path.join(dir_path, file))


def read_xml(xml_file_path):
    # XML Parsing
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    _, site, _ = xml_file_path.split("\\")
    topic = site.split(".")[0]

    # CSV Writing
    csv_file_path = os.path.join('./data/stackechange_csv', f'{site}-posts.csv')
    with open(csv_file_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ATTRIBUTES)
        for post_row in root:
            post_dict = get_post_attributes(post_row, topic)
            values = [normalize_string(v) for v in post_dict.values()]
            writer.writerow(values)


def write_csv(site: str, rows: List[Dict[str, str]]):
    file_path = os.path.join('./data/stackechange_csv', f'{site}-posts.csv')
    with open(file_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ATTRIBUTES)
        for row in rows:
            values = [normalize_string(v) for v in row.values()]
            writer.writerow(values)
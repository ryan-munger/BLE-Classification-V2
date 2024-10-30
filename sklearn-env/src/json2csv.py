import pandas as pd
from scapy import *
import json
import csv



with open('jsonfile.json', encoding='utf-8') as inputfile:
    df = pd.read_json(inputfile)

df.to_csv('csvfile.csv', encoding='utf-8', index=False)
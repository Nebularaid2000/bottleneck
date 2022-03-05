import yaml

with open('config.yaml', 'r', encoding='UTF-8') as f:
    config = yaml.safe_load(f.read())
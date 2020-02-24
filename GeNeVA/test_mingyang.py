from geneva.utils.config import keys, parse_config
import pprint
import json

cfg = parse_config()
dict_codraw_setting = vars(cfg)

with open('example_args/codraw_args.json', 'w') as fp:
    json.dump(dict_codraw_setting, fp, indent=4)

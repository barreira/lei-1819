import os
import re

root_dir = 'logs'

for subdir, dirs, files in os.walk(root_dir):
    for f in files:
        if ':' in f:
            new_name = re.sub(r'(\d{4})-(\d{2})-(\d{2})(.*)', r'\1\2\3\4', f)
            new_name = new_name.replace("_", "-")
            new_name = new_name.replace(":", "_")
            os.rename(os.path.join(subdir, f), os.path.join(subdir, new_name))

## Data Preparation

We assume that all 5 .sav files with data (student, teacher, parents, teacher self-questionnaire, principal) with no changes to file names after download.

Download link:
https://www.oecd.org/en/data/datasets/SSES-Round-1-Database.html#codebooksare

All files should be in one folder.
You can change path to that folder in `Utils.py` file:
```python
import os

def get_data_path():
    path = os.path.abspath("absolute/path/to/data")
    return path
```

Then files from 00 to 06 should be executed,
then `09_tree_models.py` has code used for charts and conclusions related to Decision Trees.

### Beware!

<u>Files generated</u> by the files from 00 to 06 must be located in the same directory as `09_tree_models.py`.



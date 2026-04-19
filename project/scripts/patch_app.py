import sys

with open("project/app.py", "r") as f:
    app_code = f.read()

import_patch = """
import sys
from project.models import model, adaboost_model, decision_tree_model, linear_model, mlp

# Alias modules so pickle can find them under the old top-level names
sys.modules['model'] = model
sys.modules['adaboost_model'] = adaboost_model
sys.modules['decision_tree_model'] = decision_tree_model
sys.modules['linear_model'] = linear_model
sys.modules['mlp'] = mlp

"""

if "sys.modules['linear_model']" not in app_code:
    app_code = app_code.replace("import os", "import os\n" + import_patch)
    # also try inserting after "import sys" if "import os" is not there
    if "import sys\n" in app_code:
        app_code = app_code.replace("import sys\n", "import sys\n" + import_patch)
    
    with open("project/app.py", "w") as f:
        f.write(app_code)
    print("Patched app.py with sys.modules aliases")

import glob

for test_file in glob.glob("tests/*.py"):
    with open(test_file, "r") as f:
        code = f.read()
    code = code.replace("from decision_tree_model", "from models.decision_tree_model")
    code = code.replace("from adaboost_model", "from models.adaboost_model")
    with open(test_file, "w") as f:
        f.write(code)

print("Tests patched")

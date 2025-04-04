import os

def list_files(startpath, max_depth=2):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        if level > max_depth:
            continue
        indent = '│   ' * level + '├── '
        print(f"{indent}{os.path.basename(root)}/")
        subindent = '│   ' * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

list_files('.', max_depth=2)

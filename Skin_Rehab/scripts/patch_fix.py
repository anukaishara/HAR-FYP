import os
import chumpy
import inspect

# 1. LOCATE THE TARGET
# This finds where the 'chumpy' library is installed on your Vivobook
chumpy_path = os.path.dirname(chumpy.__file__)
print(f"Found chumpy at: {chumpy_path}")

# 2. FIX THE 'INSPECT' ERROR (ch.py)
# Modern Python replaced getargspec with getfullargspec
ch_file = os.path.join(chumpy_path, 'ch.py')
with open(ch_file, 'r') as f:
    content = f.read()

if 'inspect.getargspec' in content:
    print("Fixing inspect.getargspec in ch.py...")
    content = content.replace('inspect.getargspec(func)', 'inspect.getfullargspec(func)')
    with open(ch_file, 'w') as f:
        f.write(content)
else:
    print("ch.py is already patched for inspect.")

# 3. FIX THE 'NUMPY' ATTRIBUTE ERROR (__init__.py)
# NumPy 1.20+ removed np.bool, np.int, etc. We must map them back manually.
init_file = os.path.join(chumpy_path, '__init__.py')
with open(init_file, 'r') as f:
    lines = f.readlines()

new_lines = []
patched_numpy = False
for line in lines:
    if "from numpy import bool, int, float, complex, object, unicode, str, nan, inf" in line:
        new_lines.append("from numpy import nan, inf\n")
        new_lines.append("bool, int, float, complex, object, unicode, str = bool, int, float, complex, object, str, str\n")
        patched_numpy = True
    else:
        new_lines.append(line)

if patched_numpy:
    print("Fixing numpy legacy types in __init__.py...")
    with open(init_file, 'w') as f:
        f.writelines(new_lines)
else:
    print("__init__.py is already patched or format differs.")

print("\n✅ SUCCESS: Chumpy library is now compatible with your system.")
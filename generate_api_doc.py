import inspect
import io
import os
import importlib

doc_path = 'docs/src/'
lib_path = 'SafeRLBench'


def generate_class_doc_file(path, name, mod_name):
    file_name = path + name + '.rst'

    if os.path.exists(file_name):
        print('already exists: ' + file_name)
    else:
        print('write for ' + file_name)
        with io.FileIO(file_name, 'w') as file:
            file.write((name + '\n').encode('utf-8'))
            file.write(('-' * len(name) + '\n\n').encode('utf-8'))
            file.write(('.. autoclass:: ' + mod_name + '.'
                        + name + '\n').encode('utf-8'))
            file.write(('   :members:').encode('utf-8'))


def generate_module_doc_file(path):
    with io.FileIO(doc_path + root + '.rst', 'w') as file:
        file.write((name + '\n').encode('utf-8'))
        file.write(('=' * len(name) + '\n\n').encode('utf-8'))
        file.write(('Description\n').encode('utf-8'))
        file.write(('-----------\n\n...\n\n').encode('utf-8'))
        if dirs:
            file.write(('Submodules\n').encode('utf-8'))
            file.write(('----------\n\n').encode('utf-8'))
            file.write(('.. toctree::\n').encode('utf-8'))
            file.write(('   :glob:\n').encode('utf-8'))
            file.write(('   :maxdepth: 1\n\n').encode('utf-8'))
            file.write(('   ' + name + '/*\n\n').encode('utf-8'))
        file.write(('Classes\n').encode('utf-8'))
        file.write(('-------\n\n').encode('utf-8'))
        file.write(('.. toctree::\n').encode('utf-8'))
        file.write(('   :glob:\n\n').encode('utf-8'))
        file.write(('   ' + name + '.classes/*').encode('utf-8'))


def get_classes(mod):
    classes = []
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj):
            classes.append(name)
    return classes


for root, dirs, files in os.walk(lib_path):
    root_dir, root_name = os.path.split(root)
    if not root_name.startswith('_'):
        name = root_name
        dirs = [d for d in dirs if not d.startswith('_')]
        files = [f for f in files if not f.startswith('_')]

        print(root)
        mod_path = doc_path + root
        if dirs and not os.path.exists(mod_path):
            os.makedirs(mod_path)

        class_path = doc_path + root + '.classes/'
        if files and not os.path.exists(class_path):
            os.makedirs(class_path)

        mod_name = root.replace('/', '.')

        print('Writing module classes for: ' + mod_name)
        i = importlib.import_module(mod_name)
        classes = get_classes(i)
        for c_name in classes:
            generate_class_doc_file(class_path, c_name, mod_name)

        file_name = doc_path + root + '.rst'
        if os.path.exists(file_name):
            'Module file with file_name already exists.'
        else:
            print('Writing module file for: ' + root.replace('/', '.'))
            generate_module_doc_file(file_name)

import os

ignores = ['node_modules']
def walk(path):
    for fn in os.listdir(path):
        if fn in ignores:
            continue
        sub = os.path.join(path, fn)
        if os.path.isdir(sub):
            for p in walk(sub):
                yield p
        elif os.path.isfile(sub):
            yield sub

def table_should_be_aligned(fn, lines):
    in_code = False
    bar_count = 0
    line_length = 0
    for idx, line in enumerate(lines):

        if line.count('```'):
            in_code = not in_code
            continue
        if in_code:
            continue

        lineno = idx + 1
        if not line.startswith('|'):
            bar_count = 0
            continue
        if not bar_count:
            line_length = len(line)
            bar_count = line.count('|')
        else:
            assert len(line) == line_length, \
                f'Table lines should be aligned, ' \
                f'expect exact line length {fn}:{lineno}'
            assert line.count('|') == bar_count, \
                f'Vertical bar count error in {fn}:{lineno}'

def check_title_syntax(path, lines):
    got_one = False
    for line in lines:
        line = line.strip(' \n')
        if line.startswith('#'):
            assert got_one or line.count('#') == 1, \
                f'First title should be top level with single # in {path}'
            assert line.count('#') < 3, \
                f'Please don\'t have ### level title in {path}'
            assert not got_one or line.count('#') > 1, \
                f'Should have ONLY one title with single # in {path}'
            got_one = True
    assert got_one, \
        f'Should have a title with single # in {path}'

must_include_titles = set([
    'Description',
    'Model',
    'References',
    'License'])
allowed_titles = set([
    'Description',
    'Model',
    'Preprocessing',
    'Postprocessing',
    'Dataset',
    'References',
    'Contributors',
    'License'])
def check_title_content(path, lines):
    includes = set()
    for idx, line in enumerate(lines):
        lineno = idx + 1
        line = line.strip(' \n')
        if not line.startswith('#'):
            continue
        if line.count('#') == 1:
            # Model name
            continue
        title = line.replace('#', '').strip()
        assert title in allowed_titles, \
            f'Please don\'t use "{title}" title in markdown, {path}:{lineno}'
        if title in must_include_titles:
            includes.add(title)
    if len(includes) != len(must_include_titles):
        lack = ', '.join(t for t in must_include_titles if t not in includes)
        print(f'Please add {lack} in {path}')
        raise Exception('Lack certain title')

def should_have_one_table_in_model_chapter(path, lines):
    table_count = 0
    in_table = False
    in_chapter = False
    for idx, line in enumerate(lines):
        lineno = idx + 1
        line = line.strip(' \n')
        if line.startswith('#'):
            title = line.replace('#', '').strip()
            if title == 'Model':
                in_chapter = True
            elif in_chapter:
                break
        if not in_chapter:
            continue
        if line.startswith('|'):
            if not in_table:
                in_table = True
                table_count += 1
        else:
            in_table = False

    assert table_count == 1, \
        f'Should have ONE-AND-ONLY-ONE table in Model chapter in {path}. ' \
        f'Got {table_count}.'

def link_markdown(path):
    with open(path) as f:
        lines = f.readlines()

    check_title_syntax(path, lines)
    check_title_content(path, lines)
    should_have_one_table_in_model_chapter(path, lines)
    table_should_be_aligned(path, lines)

def lint_markdowns(path):
    for fn in walk(path):
        if not fn.endswith('.md'):
            continue
        link_markdown(fn)

def main():
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(path, '../..'))

    os.chdir(path)
    lint_markdowns('vision')

if __name__ == '__main__':
    main()

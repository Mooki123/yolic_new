"""Read constants out of the released scripts without running them (they train or load weights at import).

Values are taken from the module-level assignments and calls with `ast`, so the layouts and the
tests stay tied to the exact numbers in the v0-released tree.
"""
import ast


def _tree(path):
    with open(path, encoding='utf-8') as f:
        return ast.parse(f.read(), filename=path)


def assigned_value(path, name, env=None):
    """Evaluate the last module-level `name = <expr>`. env supplies names the expression uses."""
    node = None
    for stmt in _tree(path).body:
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
            node = stmt.value
    if node is None:
        raise KeyError(f'{name} is not assigned in {path}')
    return eval(compile(ast.Expression(node), path, 'eval'), {'__builtins__': {}}, dict(env or {}))


def flip_tables(path):
    """Literal seq_list arguments of every random_augmentation(...) call in the file."""
    tables = []
    for node in ast.walk(_tree(path)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'random_augmentation' and len(node.args) == 3
                and isinstance(node.args[2], ast.List)):
            tables.append(ast.literal_eval(node.args[2]))
    return tables


def cityscapes_cells(path='cityscapes_yolic.py'):
    return assigned_value(path, 'cell_list')


def outdoor_cells(path='outdoor_pred.py'):
    points = assigned_value(path, 'points_list')
    return assigned_value(path, 'cell_list', {'points_list': points})


def indoor_polygons(path='indoor_pred.py'):
    return assigned_value(path, 'polygonList')

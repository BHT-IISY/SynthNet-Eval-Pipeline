from typing import Any, List, Tuple
import re

def parse_tuple_to_types(tup: Tuple[Any, Any], types: Tuple[Any, Any]) -> Tuple[Any, Any]:
    res = []
    for i in range(2):
        try:
            item = types[i](tup[i])
            res.append(item)
        except ValueError as e:
            print(f'Error parsing value: {tup[i]} with type: {types[i].__name__}.')
            raise e
    return tuple(res)

def parse_tuple_params(params: List[str], types: Tuple[Any, Any], delimiter: str = ":") -> List[Tuple[Any, Any]]:
    params = params.copy()
    # Using re.match over a simple "in" statement to exclude multiple uses of the delimiter and one side being empty
    is_delimiter_not_in_params = [ re.match(f'^\w+{delimiter}\w+$', item) is None for item in params]
    if any(is_delimiter_not_in_params):
        faulty_params = []
        while any(is_delimiter_not_in_params):
            idx = is_delimiter_not_in_params.index(True)
            faulty_params.append(params.pop(idx))
            is_delimiter_not_in_params.pop(idx)
        raise ValueError(f'Error parsing parameter key-value-pairs with delimiter: "{delimiter}". Faulty parameters are: [{", ".join(faulty_params)}].')
    parsed = [ (item[0], item[1]) for item in [ param.split(delimiter) for param in params ] ]
    splitted = [ param.split(delimiter) for param in params ]
    parsed = []
    for item in splitted:
        parsed_item = parse_tuple_to_types(item, types)
        parsed.append(parsed_item)
    return parsed

def assert_params_not_in_list(params: list, the_list: list) -> None:
    are_params_in_list = [ param in the_list for param in params]
    if all(are_params_in_list):
        return
    else:
        invalid_items = [ param for valid, param in zip(are_params_in_list, params) if not valid ]
        raise ValueError(f'Parameters {invalid_items} are not in the list of acceptable values -> {the_list}.')

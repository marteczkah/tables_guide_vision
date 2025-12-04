CAD_ATTRIBUTES_PRE = [
    "I20_premr", "I21v2_premr", "I24_premr", "I25_premr"
]

NUMERICAL_ATTRIBUTES = [
    'LVEDV', 'LVESV', 'LVSV', 'LVEF', 'LVEDM', 'LVCO', 'RVEDV',
    'RVESV', 'RVSV', 'RVEF', 'RVCO', 'MYOEDV', 'MYOESV', 'age'
]

CATEGORICAL_ATTRIBUTES = [
    'I20_postmr', 'I20_premr', 'I21v2_postmr', 'I21v2_premr', 'I24_postmr', 
    'I24_premr', 'I25_postmr', 'I25_premr', 'sex', 'smoking'
]

ATTRIBUTES = CATEGORICAL_ATTRIBUTES + NUMERICAL_ATTRIBUTES

CAT_LABELS = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 4
]

NUMERICAL_MAPPING = {
    'smoking' : {
        0: 0,
        1 : 1,
        2 : 2,
        4 : 3,
        -3: 3
    }, 
}
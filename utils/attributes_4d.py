CAD_ATTRIBUTES_PRE = [
    "I20_premr", "I21v2_premr", "I24_premr", "I25_premr"
]

NUMERICAL_ATTRIBUTES = [
    'LVEDV', 'LVESV', 'LVSV', 'LVEF', 'LVEDM', 'LVCO', 'RVEDV',
    'RVESV', 'RVSV', 'RVEF', 'RVCO', 'MYOEDV', 'MYOESV', 'age',
    'Father’s age at death','Mother’s age at death', 'Average heart rate',
    'Body surface area', 'Cardiac index', 'Cardiac operations performed', 
    'Diastolic blood pressure automated', 'Diastolic blood pressure manual reading', 'Duration of moderate activity',
    'Systolic blood pressure', 'Systolic brachial blood pressure during PWA', 'Total mass' 
]

CATEGORICAL_ATTRIBUTES = [
    'I20_postmr', 'I20_premr', 'I21v2_postmr', 'I21v2_premr', 'I24_postmr', 
    'I24_premr', 'I25_postmr', 'I25_premr', 'sex', 'smoking', 'Father still alive',
    'Mother still alive', 'Siblings have high blood pressure',
    'Father has heart disease', 'Father has high blood pressure',
    'Siblings have heart disease', 'Mother has high blood pressure',
    'Mother has heart disease', 'CAD_risk'
]

ATTRIBUTES = CATEGORICAL_ATTRIBUTES + NUMERICAL_ATTRIBUTES

CAT_LABELS = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 2,2,2,2,2,2,2
]

NUMERICAL_MAPPING = {
    'Diabetes' : {
        1 : 1,
        0 : 0,
        -1 : 2,
        -3 : 3
    },
    'CAD_risk': {
        1 : 1,
        0 : 0
    }, 
    'Siblings have heart disease': {
        1 : 1,
        0 : 0
    }, 
    'Siblings have high blood pressure' : {
        1 : 1,
        0 : 0
    },
    'Mother has heart disease' : {
        1 : 1,
        0 : 0
    },
    'Mother has high blood pressure' : {
        1 : 1,
        0 : 0
    },
    'Father has heart disease' : {
        1 : 1,
        0 : 0
    },
    'Father has high blood pressure' : {
        1 : 1,
        0 : 0
    },
    'Blood pressure medication' : {
        1 : 1,
        0 : 0
    }, 
    'Cholesterol lowering medication' : {
        1 : 1,
        0 : 0
    },
    'Diabetes' : {
        1 : 1,
        0 : 0,
        -1 : 2,
        -3 : 3
    },
    'Father still alive' : {
        1 : 1,
        0 : 0,
        -1 : 2,
        -3 : 3
    },
    'Mother still alive' : {
        1 : 1,
        0 : 0,
        -1 : 2,
        -3 : 3
    }, 
    'Non-accidental death in close genetic family' : {
        1 : 1,
        0 : 0,
        -1 : 2,
        -3 : 3
    }, 
    'Alcohol drinker status' : {
        -3: 3,
        0: 0,
        1: 1,
        2: 2
    }, 
    'Alcohol intake frequency': {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
       -3: 0,
    },
    'Alcohol usually taken with meals' : {
        1 : 1,
        0: 0,
        -6: 2,
        -3: 3,
        -1 : 4
    },
    'Amount of alcohol drunk on a typical drinking day' : {
        -818: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        0: 0, # NaN
    }, 
    'Beef intake' : {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        -1: 6,
        -3: 7
    }, 
    'Hormone replacement therapy medication regularly taken' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        -7: 6,
        -1: 0,
        -3: 7
    }, 
    'Cholesterol lowering medication regularly taken' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        -7: 4,
        -1: 0,
        -3: 5
    }, 
    'Current tobacco smoking' : {
        1: 1,
        2: 2,
        0: 0,
        -3: 3
    }, 
    'Duration of heavy DIY' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        -1: 0,
        -3: 8
    }, 
    'Duration of light DIY' :  {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        -1: 0,
        -3: 8
    }, 
    'Duration of other exercises' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        -1: 0,
        -3: 8
    }, 
    'Duration of strenuous sports' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        -1: 0,
        -3: 8
    }, 
    'Duration walking for pleasure' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        -1: 0,
        -3: 8
    }, 
    'Ever smoked' : {
        0 : 0,
        1 : 1
    }, 
    'Falls in the last year' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        -3: 0
    }, 
    'Frequency of consuming six or more units of alcohol' : {
        0: 0, # NaN
        -818: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    }, 
    'Frequency of drinking alcohol' : {
        0: 0, # NaN
        -818: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    }, 
    'Frequency of heavy DIY in last 4 weeks' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        -1: 0,
        -3: 7
    }, 
    'Frequency of other exercises in last 4 weeks' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        -1: 0,
        -3: 7
    }, 
    'Frequency of stair climbing in last 4 weeks' : {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        -1: 6,
        -3: 7
    },
    'Frequency of strenuous sports in last 4 weeks' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        -1: 0,
        -3: 7
    }, 
    'Frequency of walking for pleasure in last 4 weeks' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        -1: 0,
        -3: 7
    }, 
    'Lamb/mutton intake' : {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        -1: 6,
        -3: 7
    },
    'Days/week moderate activity' : {
        0 : 0,
        1: 1,
        2: 2,
        3: 3, 
        4: 4,
        5: 5,
        6 : 6,
        7 : 7,
        -3: 8,
        -1 : 9
    }, 
    'Days/week vigorous activity' : {
        0 : 0,
        1: 1,
        2: 2,
        3: 3, 
        4: 4,
        5: 5,
        6 : 6,
        7 : 7,
        -3: 8,
        -1 : 9
    }, 
    'Overall health rating' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        -1: 0,
        -3: 5
    }, 
    'Pace' : {
        0: 0,
        1 : 1
    }, 
    'Past tobacco smoking' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        -3: 0
    }, 
    'Pork intake'  : {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        -1: 6,
        -3: 7
    },
    'Processed meat intake'  : {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        -1: 6,
        -3: 7
    }, 
    'Shortness of breath walking on level ground' : {
        1: 1,
        0 : 0,
        -1 : 2,
        -3 : 3
    }, 
    'Sleeplessness / insomnia' : {
        0: 0, # NaN
        1 : 1,
        2 : 2,
        3: 3,
        -3 : 0
    }, 
    'Smoking/smokers in household' : {
        1: 1,
        2: 2,
        0: 0,
        -3: 3
    }, 
    'Tense / highly strung': {
        1: 1,
        0 : 0,
        -1 : 2,
        -3 : 3
    },
    'Usual walking pace' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        -7: 4,
        -3: 0
    }, 
    'Weight change compared with 1 year ago' : {
        0: 0,
        2: 1,
        3: 2,
        -1: 3,
        -3: 4
    }, 
    'Worrier / anxious feelings' : {
        1: 1,
        0 : 0,
        -1 : 2,
        -3 : 3
    }, 
    'smoking' : {
        0: 0,
        1 : 1,
        2 : 2,
        4 : 3,
        -3: 3
    }, 
    'sex' : {
        0: 0,
        1: 1
    }, 
    'I20_postmr' : {
        0: 0,
        1: 1
    }, 
    'I20_premr' : {
        0: 0,
        1: 1
    }, 
    'I21v2_postmr': {
        0: 0,
        1: 1
    }, 
    'I21v2_premr': {
        0: 0,
        1: 1
    }, 
    'I24_postmr': {
        0: 0,
        1: 1
    }, 
    'I24_premr': {
        0: 0,
        1: 1
    }, 
    'I25_postmr': {
        0: 0,
        1: 1
    }, 
    'I25_premr': {
        0: 0,
        1: 1
    }, 
    'I50_premr': {
        0: 0,
        1: 1
    }, 
    'I50_postmr': {
        0: 0,
        1: 1
    }, 
    'I48_premr': {
        0: 0,
        1: 1
    }, 
    'I48_postmr': {
        0: 0,
        1: 1
    }, 
    'I35_premr': {
        0: 0,
        1: 1
    }, 
    'I35_postmr': {
        0: 0,
        1: 1
    }, 
    'I31_premr': {
        0: 0,
        1: 1
    }, 
    'I31_postmr': {
        0: 0,
        1: 1
    }, 
    'I47_premr': {
        0: 0,
        1: 1
    }, 
    'I47_postmr': {
        0: 0,
        1: 1
    }, 
    'I30_premr': {
        0: 0,
        1: 1
    }, 
    'I30_postmr': {
        0: 0,
        1: 1
    }, 
    'I44_premr': {
        0: 0,
        1: 1
    }, 
    'I44_postmr': {
        0: 0,
        1: 1
    }, 
    'I34_premr': {
        0: 0,
        1: 1
    }, 
    'I34_postmr': {
        0: 0,
        1: 1
    }, 
    'I33_premr': {
        0: 0,
        1: 1
    }, 
    'I33_postmr': {
        0: 0,
        1: 1
    }, 
    'I49_premr': {
        0: 0,
        1: 1
    }, 
    'I49_postmr': {
        0: 0,
        1: 1
    }, 
    'I42_premr': {
        0: 0,
        1: 1
    }, 
    'I42_postmr': {
        0: 0,
        1: 1
    }, 
    'I40_premr': {
        0: 0,
        1: 1
    }, 
    'I40_postmr': {
        0: 0,
        1: 1
    }, 
    'I45_premr': {
        0: 0,
        1: 1
    }, 
    'I45_postmr': {
        0: 0,
        1: 1
    }, 
    'I46_premr': {
        0: 0,
        1: 1
    }, 
    'I46_postmr': {
        0: 0,
        1: 1
    }, 
    'I36_premr': {
        0: 0,
        1: 1
    }, 
    'I36_postmr': {
        0: 0,
        1: 1
    }, 
    'I10_premr': {
        0: 0,
        1: 1
    }, 
    'I10_postmr': {
        0: 0,
        1: 1
    }, 
    'I11_premr': {
        0: 0,
        1: 1
    }, 
    'I11_postmr': {
        0: 0,
        1: 1
    }, 
    'I12_premr': {
        0: 0,
        1: 1
    }, 
    'I12_postmr': {
        0: 0,
        1: 1
    }, 
    'I13_premr': {
        0: 0,
        1: 1
    }, 
    'I13_postmr': {
        0: 0,
        1: 1
    }, 
    'I15_premr': {
        0: 0,
        1: 1
    }, 
    'I15_postmr': {
        0: 0,
        1: 1
    }, 
    'E10_premr': {
        0: 0,
        1: 1
    }, 
    'E10_postmr': {
        0: 0,
        1: 1
    }, 
    'E11_premr': {
        0: 0,
        1: 1
    }, 
    'E11_postmr': {
        0: 0,
        1: 1
    }, 
    'E13_premr': {
        0: 0,
        1: 1
    }, 
    'E13_postmr': {
        0: 0,
        1: 1
    }, 
    'E14_premr': {
        0: 0,
        1: 1
    }, 
    'E14_postmr': {
        0: 0,
        1: 1
    },
}
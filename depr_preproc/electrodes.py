electrode_mapping = {
    'E36': 'C3',
    'E104': 'C4',
    'E129': 'Cz',
    'E24': 'F3',
    'E124': 'F4',
    'E33': 'F7',
    'E122': 'F8',
    'E22': 'Fp1',
    'E9': 'Fp2',
    'E11': 'Fz',
    'E70': 'O1',
    'E83': 'O2',
    'E52': 'P3',
    'E92': 'P4',
    'E58': 'T5',
    'E96': 'T6',
    'E45': 'T3',
    'E108': 'T4',
    'E62': 'Pz'
}

channels_to_use = [
    # prefrontal
    'Fp1',
    'Fp2',
    # frontal
    'F7',
    'F3',
    'F4',
    'Fz',
    'F8',
    # central and temporal
    'T7',
    'C3',
    #'Cz',
    'C4',
    'T8',
    # parietal
    'P7',
    'P3',
    'Pz',
    'P4',
    'P8',
    # occipital
    'O1',
    'O2'
]
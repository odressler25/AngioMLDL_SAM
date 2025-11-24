import pandas as pd
import json

df = pd.read_csv('E:/AngioMLDL_data/corrected_dataset_training.csv')

print('Sample view angles from first 10 cases:\n')
for i in range(10):
    jpath = df['contours_path'].iloc[i]
    jdata = json.load(open(jpath))
    angles = jdata.get('view_angles', {})
    cine_name = df['cine_path'].iloc[i].split('/')[-1].replace('_cine.npy', '')
    primary = angles.get('primary_angle', 0)
    secondary = angles.get('secondary_angle', 0)
    print(f'{cine_name}:')
    print(f'  Primary={primary:.1f}, Secondary={secondary:.1f}')

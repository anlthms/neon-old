import os
from glob import glob
import gzip

TOPDIR = '/usr/local/data/MULTIPIE'
OUTDIR = TOPDIR + '/batch_data'
CAMS = ['11_0', '12_0', '09_0', '08_0', '13_0', '14_0', '051', '05_0', '041',
        '19_0', '20_0', '01_0', '24_0']
CAMPOS = ['110', '120', '090', '080', '130', '140', '051', '050', '041', '190',
          '200', '010', '240']
CAMDICT = {CAMPOS[i]: i for i in range(len(CAMPOS))}
EXPDICT = {'0101': 0, '0102': 1, '0201': 0, '0202': 2, '0203': 3, '0301': 0,
           '0302': 1, '0303': 4, '0401': 0, '0402': 0, '0403': 5}
EXPLIST = ['neutral', 'smile', 'surprise', 'squint', 'disgust', 'scream']

FRONTALONLY = False

"""
We're going to label
    subject (l_sub), camera (l_cam), expression (l_exp), illumination (l_ill)

Expression Mappings (SESSION_RECORDING to Expression)
0, neutral : 01_01, 02_01, 03_01, 04_01, 04_02,
1, smile   : 01_02, 03_02
2, surprise: 02_02
3, squint  : 02_03
4, disgust : 03_03
5, scream  : 04_03
"""

if FRONTALONLY:
    OUTDIR = TOPDIR + '-frontal/batch_data'
    CAMS = ['14_0', '051', '05_0']
    CAMPOS = ['140', '051', '050']
    CAMDICT = {CAMPOS[i]: i for i in range(len(CAMPOS))}


def parse_filename(fname):
    f = os.path.basename(fname)[:-4].split('_')
    return (fname, int(f[0]), CAMDICT[f[3]], EXPDICT[f[1]+f[2]], int(f[-1]))

search_path = os.path.join(TOPDIR, 'session*/multiview/*/*')
files_per_cam = {cc: glob(os.path.join(search_path, cc, '*')) for cc in CAMS}
csv_per_cam = {cc: map(parse_filename, files_per_cam[cc]) for cc in CAMS}

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

trainfile = os.path.join(OUTDIR, 'train_file.csv.gz')
valfile = os.path.join(OUTDIR, 'val_file.csv')

with gzip.open(trainfile, 'wb') as f:
    f.write('filename,l_sub,l_cam,l_exp,l_ill\n')
    for cc in CAMS:
        for tup in csv_per_cam[cc]:
            f.write('{},{},{},{},{}\n'.format(*tup))
f.close()

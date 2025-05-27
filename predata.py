

import os
import argparse
import pandas as pd
import numpy as np
import shutil

"""
This script is intended to serve the purpose of transforming raw data (extracted from an SQL database fed by an annotation web API) into something that can be fed 
to an algorithm to be able to extract features
Reads patches and labels (from validation and dense training set, recall that these images are annotated by all markers of the 
crowdsourced dataset) to train a CNN for feature extraction (these images are the ones that are annotated by all residents)
and ultimately, classificate the image
"""



def data_curation(args): #translate tables of annotations into objects in python. Move desired images/files to folders
    """
    We extract annotations for validation and dense set
    """

    dict_markers_val = pd.read_excel(os.path.join(args.data_dir, 'Data_patches.xls'),
                                   sheet_name=['marker1', 'marker2', 'marker3', 'marker4', 'marker5', 'marker6', 'marker7'])
    dict_markers_dense = pd.read_excel(os.path.join(args.data_dir, 'Data_patches_DT.xls'),
                                     sheet_name=['marker1', 'marker2', 'marker3', 'marker4', 'marker5', 'marker6', 'marker7'])


    df_markers_val, val_df = data_creation(dict_markers_val, 'val')
    df_markers_dense, dense_df = data_creation(dict_markers_dense, 'dense')

    #define aggregation of labels by majority voting of 3 or more markers (maj_3), 4 or more markers (maj_4) or ground
    #truth (g_t)
    gt_df_val, maj_df_val, list_val = majority_voting(args, df_markers_val, val_df)
    gt_df_dense, maj_df_dense, list_dense = majority_voting(args, df_markers_dense, dense_df)

    #move patches to correct folder structure for later working with data generators
    move_patches(args, gt_df_val, maj_df_val, 'val' )
    move_patches(args, gt_df_dense, maj_df_dense, 'dense' )

def data_creation(data_df, partition):

    ddf = {}
    markers_df = pd.DataFrame()

    for i in range(len(data_df)):
        i += 1
        marker = 'marker{s}'.format(s=str(i))
        marker_df = data_df.get(marker)
        marker_df = marker_df.dropna(axis=1)
        markers = [marker] * len(marker_df)
        marker_df.loc[:, 'markers'] = markers
        markers_df = pd.concat([markers_df, marker_df])
        ddf['marker{s}'.format(s=str(i))] = marker_df #DICTIONARY OF DATAFRAMES

    label = ['Grade 3+3', 'Grade 3+4', 'Grade 3+5', 'Grade 4+3', 'Grade 4+4', 'Grade 4+5', 'Grade 5+3', 'Grade 5+4',
             'Grade 5+5', 'Grade 5', 'Grade 4', 'Grade 3', 'Bening (healthy)']
    eti = ['Grado 3+3', 'Grado 3+4', 'Grado 3+5', 'Grado 4+3', 'Grado 4+4', 'Grado 4+5', 'Grado 5+3', 'Grado 5+4',
           'Grado 5+5', 'Grado 5', 'Grado 4', 'Grado 3', 'Bening (saludable)']
    labels = eti + label

    #we extract only the images which labels correspond to a gleason score
    # this is a dataframe of all the patches annotated by all markers in 'Annotation1' column --> we are ignoring
    # the secondary annotations
    markers_df = markers_df[markers_df['Annotation 1'].isin(labels)] #for keeping track
    markers_df_maj = markers_df.drop(['Index', 'Image #'], axis=1).reset_index(drop=True) #for working


    values = markers_df['Patch filename'].value_counts()
    #only extracting patches not annotated by 4 or more residents as Other ???????
    names = values[values.values >= 4].index.tolist()

    markers_fin = pd.DataFrame()
    for k in range(len(names)):
        name = names[k]
        markers_df_fin = markers_df[markers_df['Patch filename'].isin([name])]

        anotaciones = markers_df_fin['Annotation 1'].tolist()
        anotadores = markers_df_fin['markers'].tolist()

        markers_df_fin = markers_df_fin.drop(['Index', 'Image #'], axis=1).reset_index(drop=True)
        markers_df_fin = markers_df_fin.transpose()
        #col = markers_df_fin.columns

        markers_df_fin.rename(columns={0: 'marker1', 1: 'marker2', 2: 'marker3', 3: 'marker4', 4: 'marker5', 5: 'marker6', 6: 'marker7'}, inplace=True)
        markers_df_fin = markers_df_fin.reset_index(drop=True)
        #markers_df_fin = markers_df_fin.drop([0, 1, 2, 4], axis=0, inplace=True)
        #markers_df_fin_row = pd.DataFrame(list(zip(patches, anotaciones, anotadores)), columns=['Patch name', 'Labels', 'Markers'])

        markers_df_fin = markers_df_fin.drop([0, 2], axis=0).reset_index(drop=True)
        markers_df_fin['Patch filename'] = name
        markers_fin = pd.concat([markers_df_fin, markers_fin])
        markers_fin = markers_fin.reset_index(drop=True)

    if partition == 'val':
        markers_fin['Patch filename'] = markers_fin['Patch filename'].str.split(pat='V_', expand=True)[1]
        markers_df_maj['Patch filename'] = markers_df_maj['Patch filename'].str.split(pat='V_', expand=True)[1]
    else:
        markers_fin['Patch filename'] = markers_fin['Patch filename'].str.split(pat='DT_', expand=True)[1]
        markers_df_maj['Patch filename'] = markers_df_maj['Patch filename'].str.split(pat='DT_', expand=True)[1]
    markers_fin['Patch filename'] = markers_fin['Patch filename'].str.split(pat='.', expand=True)[0]
    markers_df_maj['Patch filename'] = markers_df_maj['Patch filename'].str.split(pat='.', expand=True)[0]
    markers_fin = markers_fin.dropna(axis=0).reset_index(drop=True)


    #adding ground truth (validation and dense training sets)
    #no dense training set not used for feature extraction (to validate that model) = greater uncertainty
    labels_ann = pd.read_csv('...csv')
    nomes = markers_fin['Patch filename'].tolist()
    markers_fin_df = pd.DataFrame()

    for nom in range(len(nomes)):
        label = labels_ann[labels_ann['image_name'].isin([nomes[nom]])]['label']
        if label.shape[0] == 0:
            label = '0'
        else:
            label = label.values[0]

        markers_fin_row = markers_fin[markers_fin['Patch filename'].isin([nomes[nom]])]
        markers_fin_row['ground truth'] = label
        #todo
        markers_fin_df = pd.concat([markers_fin_df, markers_fin_row])

    # normalize labels
    list_lab_3 = ['Grade 3', 'Grado 3', 'Grado 3+3', 'Grado 3+4', 'Grado 3+5', 'Grade 3+3', 'Grade 3+4', 'Grade 3+5',
                  '3+3', '3+4', '3+5']
    list_lab_4 = ['Grade 4', 'Grado 4', 'Grado 4+3', 'Grado 4+4', 'Grado 4+5', 'Grade 4+3', 'Grade 4+4', 'Grade 4+5',
                  '4+3', '4+4', '4+5']
    list_lab_5 = ['Grade 5', 'Grado 5', 'Grado 5+3', 'Grado 5+4', 'Grado 5+5', 'Grade 5+3', 'Grade 5+4', 'Grade 5+5',
                  '5+3', '5+4', '5+5']
    list_lab_0 = ['Bening (healthy)', 'Bening (saludable)', '0+0']
    markers_fin_norm = markers_fin_df.replace(list_lab_3, '3').replace(list_lab_4, '4').replace(list_lab_5, '5').replace(
        list_lab_0, '0')
    markers_df_maj_norm = markers_df_maj.replace(list_lab_3, '3').replace(list_lab_4, '4').replace(list_lab_5, '5').replace(
        list_lab_0, '0')
    return markers_fin_norm, markers_df_maj_norm


def majority_voting(args, norm_gt_df, norm_df):

    gt_df = norm_gt_df.drop(['marker1', 'marker2', 'marker3', 'marker4', 'marker5', 'marker6', 'marker7'], axis=1)
    norm_df.reset_index(drop=True)
    norm_df_list = norm_df['Patch filename'].tolist()
    maj_df = pd.DataFrame()
    for patch in range(len(norm_df_list)):
        labels = norm_df[norm_df['Patch filename'].isin([norm_df_list[patch]])]['Annotation 1']
        label = np.max(labels)

        num_ann = (labels == label).value_counts()
        if args.criteria == 'maj_3' and num_ann[True] >= 3:
            norm_gt_df_row = norm_gt_df[norm_gt_df['Patch filename'].isin([norm_df_list[patch]])]
            norm_gt_df_row['label_agg'] = label
        elif args.criteria == 'maj_4' and num_ann[True] >= 4:
            norm_gt_df_row = norm_gt_df[norm_gt_df['Patch filename'].isin([norm_df_list[patch]])]
            norm_gt_df_row['label_agg'] = label
        elif args.criteria == 'g_t':
            break
        else:
            continue
        maj_df = pd.concat([maj_df, norm_gt_df_row])

    correct = (maj_df['ground truth'] == maj_df['label_agg']).value_counts()[True]
    incorrect = (maj_df['ground truth'] == maj_df['label_agg']).value_counts()[False]
    list_comp = [correct, incorrect]

    return gt_df, maj_df, list_comp
    #return a dataframe with name of patch and ground truth, dataframe with aggregated label and the list of aggregated
    # labels that matchs and doesn´t match the ground truth or gold label by experts


def move_patches(args, data_gt, data_agg, partition):
    dir_out = '/data/Prostata/Images/Feature_Extraction/Images'
    val_dir_in = '/data/Prostata/Images/Validation/Patches'
    dense_dir_in = '/data/Prostata/Images/Training/Dense'
    os.makedirs(dir_out, exist_ok=True)
    if args.criteria == 'g_t':
        patches = data_gt['Patch filename'].tolist()
        labels = data_gt['ground truth'].tolist()
        gt_dir_out = '/data/Prostata/Images/Feature_Extraction/Images/gt_aggregation'
        os.makedirs(gt_dir_out, exist_ok=True)
        for patch in range(len(patches)):
            os.makedirs(os.path.join(gt_dir_out, str(labels[patch])), exist_ok=True)
            dst_file = os.path.join(gt_dir_out, str(labels[patch]), patches[patch] + '.jpg')
            if partition == 'val':
                src_file = os.path.join(val_dir_in, 'V_' + patches[patch] + '.jpg')
            elif partition == 'dense':
                src_file = os.path.join(dense_dir_in, patches[patch] + '.jpg')
            shutil.copy(src_file, dst_file)
    elif args.criteria == 'maj_3':
        patches = data_agg['Patch filename'].tolist()
        labels = data_agg['label_agg'].tolist()
        maj3_dir_out = '/data/Prostata/Images/Feature_Extraction/Images/maj3_aggregation'
        os.makedirs(maj3_dir_out, exist_ok=True)
        for patch in range(len(patches)):
            os.makedirs(os.path.join(maj3_dir_out, str(labels[patch])), exist_ok=True)
            dst_file = os.path.join(maj3_dir_out, str(labels[patch]), patches[patch] + '.jpg')
            if partition == 'val':
                src_file = os.path.join(val_dir_in, 'V_' + patches[patch] + '.jpg')
            elif partition == 'dense':
                src_file = os.path.join(dense_dir_in, patches[patch] + '.jpg')
            shutil.copy(src_file, dst_file)
    elif args.criteria == 'maj_4':
        patches = data_agg['Patch filename'].tolist()
        labels = data_agg['label_agg'].tolist()
        maj4_dir_out = '/data/Prostata/Images/Feature_Extraction/Images/maj4_aggregation'
        os.makedirs(maj4_dir_out, exist_ok=True)
        for patch in range(len(patches)):
            os.makedirs(os.path.join(maj4_dir_out, str(labels[patch])), exist_ok=True)
            dst_file = os.path.join(maj4_dir_out, str(labels[patch]), patches[patch] + '.jpg')
            if partition == 'val':
                src_file = os.path.join(val_dir_in, 'V_' + patches[patch] + '.jpg')
            elif partition == 'dense':
                src_file = os.path.join(dense_dir_in, patches[patch] + '.jpg')
            shutil.copy(src_file, dst_file)



def some_args():
    parser = argparse.ArgumentParser(description='Specify parameters to curate and create the folder structure of the database')
    parser.add_argument('--data_dir', '-dd', type=str, default='/data/microdraw/extractDataFromDatabase')
    parser.add_argument('--criteria', '-c', type=str, default='maj_4') #options= maj_3, maj_4, g_t
    return parser.parse_args()

def main_predata():
    args = some_args()
    print('Arguments provided_:')
    print(args)
    data_curation(args)


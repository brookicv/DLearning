import SimpleITK as sitk
import os
import json
import glob
import pandas as pd 


def dicom_metainfo(dicm_path, list_tag):
    '''
    获取dicom的元数据信息
    :param dicm_path: dicom文件地址
    :param list_tag: 标记名称列表,比如['0008|0018',]
    :return:
    '''
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(dicm_path)
    reader.ReadImageInformation()
    return [reader.GetMetaData(t) for t in list_tag]


def dicom2array(dcm_path):
    '''
    读取dicom文件并把其转化为灰度图(np.array)
    https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
    :param dcm_path: dicom文件
    :return:
    '''
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(dcm_path)
    image_file_reader.ReadImageInformation()
    image = image_file_reader.Execute()
    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
            image = sitk.InvertIntensity(image, maximum=255)
        image = sitk.Cast(image, sitk.sitkUInt8)
    img_x = sitk.GetArrayFromImage(image)[0]
    return img_x

def get_annotation_info(trainPath, jsonPath):  
    annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))  
    json_df = pd.read_json(jsonPath)  
    for idx in json_df.index:  
        studyUid = json_df.loc[idx, "studyUid"]  
        seriesUid = json_df.loc[idx, "data"][0]['seriesUid']  
        instanceUid = json_df.loc[idx, "data"][0]['instanceUid']  
        annotation = json_df.loc[idx, "data"][0]['annotation']  
        row = pd.Series(  
            {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})  
        annotation_info = annotation_info.append(row, ignore_index=True)  
    dcm_paths = glob.glob(os.path.join(trainPath, "**", "**.dcm"))  # 具体的图片路径  
    # 'studyUid','seriesUid','instanceUid'  
    tag_list = ['0020|000d', '0020|000e', '0008|0018']  
    dcm_info = pd.DataFrame(columns=('dcmPath', 'studyUid', 'seriesUid', 'instanceUid'))  
    for dcm_path in dcm_paths:  
        try:  
            studyUid, seriesUid, instanceUid = dicom_metainfo(dcm_path, tag_list) 
            row = pd.Series(  
                {'dcmPath': dcm_path, 'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid})  
            dcm_info = dcm_info.append(row, ignore_index=True)  
        except:
            continue  
    result = pd.merge(annotation_info, dcm_info, on=['studyUid', 'seriesUid', 'instanceUid'])
    result = result.set_index('dcmPath')['annotation']  # 然后把index设置为路径，值设置为annotation  
    return result
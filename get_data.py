import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

tags = tfio.image.dicom_tags.SeriesDescription

""" read txt file """
with open('dicom_files.txt','r') as f:
    paths = f.readlines()

""" open each file """
print(len(paths))
for idx, file_path in enumerate(paths):
    if idx % 1000 == 0:
        print(idx)

    ds = tf.io.read_file(file_path.rstrip('\n'))
    series_description = tfio.image.decode_dicom_data(ds, tags)
    series_description = series_description.numpy().decode("utf-8")

    if series_description.find('3D_DESS')== -1:
        pass
    else:
        text_file = open("dicom_3D_SGPR.txt", "w")
        n = text_file.write(file_path)

""" close file """
text_file.close()

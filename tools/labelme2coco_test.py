from labelme2coco import convert

# set directory that contains labelme annotations and image files
labelme_folder = r"/home/chenzy/FastInst-main/datasets/new_table_tennis"

# set export dir
export_dir = r"/home/chenzy/FastInst-main/datasets/new_table_tennis/ana"

# set train split rate
train_split_rate = 0.85

# set category ID start value
category_id_start = 0

# convert labelme annotations to coco
convert(labelme_folder, export_dir, train_split_rate, category_id_start=category_id_start)

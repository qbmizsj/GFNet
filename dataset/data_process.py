import os
import shutil
# 当前文件夹所在地址
# 第一批dti地址: '/home/zhang_istbi/zhangsj/ACGF/scz_dti_after_process'
# 第二批: /home/zhang_istbi/data_disk/zhang_istbi/chenxiang/scz_NMorphCH/DWI_proc
current_path = '/home/zhang_istbi/data_disk/zhang_istbi/chenxiang/scz_NMorphCH/DWI_proc'
# 要存入邻接矩阵和节点的地址
adj_path = '/home/zhang_istbi/zhangsj/AD_class/DTI_second/raw/adjacent_matrix'
nf_path = '/home/zhang_istbi/zhangsj/AD_class/DTI_second/raw/node_feature'
# if not os.path.exists(nofe_path):
# 	os.mkdir(nofe_path)
# 遍历当前文件夹，取出各个文件夹中的node feature和adjacent matrix
for root_dir, dir, _ in os.walk(current_path, topdown=True):
	# print(f'{root_dir}, {dir}, {info_list}\n')
    for dir_name in dir:
        nf_data_dir = dir_name + '_roi_nodevalue.csv'
        adj_data_dir = dir_name + '_result_net.csv'
        old_node_feature = os.path.join(root_dir, dir_name, nf_data_dir)
        old_adj_matrix = os.path.join(root_dir, dir_name, adj_data_dir)
        new_node_feature = os.path.join(nf_path, nf_data_dir)
        new_adj_matrix = os.path.join(adj_path, adj_data_dir)
        shutil.copyfile(old_node_feature, new_node_feature)
        shutil.copyfile(old_adj_matrix, new_adj_matrix)
	

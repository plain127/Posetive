import json
import numpy as np
import pandas as pd

class Pe2Pg:
    def __init__(self, generation_id):
        self.generation_id = generation_id
        self.ppe_path = f"/model/generation/{generation_id}/converted_target_pose.json"
        self.hpe_path = f"/model/generation/{generation_id}/converted_condition_pose.json"
        self.ppe_name = None
        self.hpe_name = None
        self.ppe_mpii_joints = []
        self.ppe_coco_joints = []
        self.hpe_mpii_joints = []
        self.hpe_coco_joints = []

    #pe json 파일 파싱
    def load_pe(self, pe_path):
        with open(pe_path, "r") as f:
            pe_data = json.load(f)
        return pe_data
    '''
    def bring_hpe(self):
        hpe_data = self.load_pe(self.hpe_path)
        self.hpe_name = hpe_data[0]['image']

        for i in range(17):
            if hpe_data[0]['joint_vis'][i] == 0:
                self.hpe_joints.append([-1,-1])
            elif hpe_data[0]['joint_vis'][i] == 1:
                self.hpe_joints.append(hpe_data[0]['joints'][i])

        self.hpe_joints = [[int(value) for value in sublist] for sublist in self.hpe_joints]
    '''
    def matching(self, pe_data):
        pe_joints = []
        for i in range(16):
            if pe_data[0]["joint_vis"][i] == 0:
                pe_joints.append([-1,-1])
            elif pe_data[0]["joint_vis"][i] == 1:
                pe_joints.append(pe_data[0]["joints"][i])
        
        pe_joints = [[int(value) for value in sublist] for sublist in pe_joints]
        return pe_joints

    def mpii_data(self):
        ppe_data = self.load_pe(self.ppe_path)
        hpe_data = self.load_pe(self.hpe_path)
        self.ppe_name = "converted_" + ppe_data[0]["image"]
        self.hpe_name = "converted_" + hpe_data[0]["image"]

        self.ppe_mpii_joints = self.matching(ppe_data)
        self.hpe_mpii_joints = self.matching(hpe_data)

    #mpii => coco
    def convert(self, mpii_joints):
        coco_joints = []

        coco_joint_name = ['Rank', 'Rkne', 'Rhip', 'Lhip', 'Lkne', 'Lank', 'remove1',
                           'remove2' ,'neck', 'nose', 'Rwri', 'Relb', 'Rsho', 'Lsho',
                           'Lelb', 'Lwri', 'Reye', 'Leye', 'Rear', 'Lear']

        mapping1 = {coco_joint_name[i] : mpii_joints[i] for i in range(16)}
        
        mapping1['nose'][1] += 10
        reye_x = float(mapping1['nose'][0] - 2)
        reye_y = float(mapping1['nose'][1] - 3)
        
        leye_x = float(mapping1['nose'][0] + 2)
        leye_y = float(mapping1['nose'][1] - 3)

        lear_x = float(mapping1['nose'][0] + 4)
        lear_y = float(mapping1['nose'][1] - 2)

        rear_x = float(mapping1['nose'][0] - 4)
        rear_y = float(mapping1['nose'][1] - 2)

        coco_order = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 
                      'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip',
                      'Lkne', 'Lank']
        
        mapping2 = {key : mapping1[key] for key in coco_order}

        for joint in list(mapping2.values()):
            coco_joints.append(joint)
        
        coco_joints.append([reye_x, reye_y])
        coco_joints.append([leye_x, leye_y])
        coco_joints.append([rear_x, rear_y])
        coco_joints.append([lear_x, lear_y])

        return coco_joints

    '''
    def hpe_order(self):
        self.bring_hpe()
        omni_order = ['nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lsho',
                      'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri', 'Lhip',
                      'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']
        mapping1 = {omni_order[i] : self.hpe_joints[i] for i in range(17)}
        
        neck_x = float(mapping1['nose'][0])
        neck_y = float(mapping1['nose'][0] + 15)

        coco_order = ['nose', 'Rsho', 'Relb', 'Rwri', 'Lsho', 
                      'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip',
                      'Lkne', 'Lank', 'Reye', 'Leye', 'Rear', 'Lear']
        
        mapping2 = {key : mapping1[key] for key in coco_order}

        self.hpe_pg2_joints.append(mapping2['nose'])
        del mapping2['nose']
        self.hpe_pg2_joints.append([neck_x, neck_y])

        for joint in list(mapping2.values()):
            self.hpe_pg2_joints.append(joint)
    '''

    #csv 역직렬화
    def make_csv_list(self, name, pe_joints):
        joints = np.array(pe_joints)
        joints_y = joints[:, 1]
        joints_x = joints[:, 0]

        input_csv = name + ': [' + ' '.join(map(str,joints_y)) + ']: [' + ' '.join(map(str,joints_x)) + ']'

        first_bracket_start = input_csv.index('[')
        first_bracket_end = input_csv.index(']')
        second_bracket_start = input_csv.index('[', first_bracket_end)
        second_bracket_end = input_csv.index(']', second_bracket_start)

        first_list = input_csv[first_bracket_start + 1:first_bracket_end].replace(' ', '\n')
        second_list = input_csv[second_bracket_start + 1:second_bracket_end].replace(' ', '\n')

        split_string = f"{name}: [{first_list}]: [{second_list}]"
        input_csv_list = split_string.split("\n")
        
        return input_csv_list

    def save_anno_csv(self):
        anno_path = f"/model/generation/{self.generation_id}/annotation.csv"
        pair_path = f"/model/generation/{self.generation_id}/pairs.csv"
        #self.hpe_order()
        self.mpii_data()
        self.hpe_coco_joints = self.convert(self.hpe_mpii_joints)
        self.ppe_coco_joints = self.convert(self.ppe_mpii_joints)

        anno_data = {}
        pair_data = {0:['from', self.hpe_name], 1:['to', self.ppe_name]}
        pair_df = pd.DataFrame(pair_data)

        df = pd.DataFrame(columns = [i for i in range(35)])
        first_row = ['name:keypoints_y:keypoints_x']

        for i in range(34):
            first_row.append(None)

        df.loc[0] = first_row

        hpe_input_csv_list = self.make_csv_list(self.hpe_name, self.hpe_coco_joints)

        for i in range(35):
            anno_data[i] = hpe_input_csv_list[i]

        df2 = pd.DataFrame(anno_data, index=[1])

        ppe_input_csv_list = self.make_csv_list(self.ppe_name, self.ppe_coco_joints)
        
        for i in range(35):
            anno_data[i] = ppe_input_csv_list[i]

        df3 = pd.DataFrame(anno_data, index=[2])
        final_df =pd.concat([df, df2, df3], ignore_index=False)

        final_df.to_csv(f"{anno_path}", index=False, header=False)
        pair_df.to_csv(f"{pair_path}", index=False, header=False)







        
        



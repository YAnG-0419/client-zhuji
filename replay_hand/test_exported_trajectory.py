import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from build_exported_trajectory import convert,model_names
from apply_common_placement import apply


class ExportTests(unittest.TestCase):
    def fixture(self,root):
        c=root/'config';(c/'calibration').mkdir(parents=True)
        cfg=dict(board_center_in_board_m=[.14,.09,0],sides={})
        payload=dict(time_s=np.array([0.,1/30]),timestamp_ns=np.array([1000000000,1033333333]),
                     frame='lychee_root',quaternion_order='wxyz',position_unit='metre',joint_unit='radian',
                     alignment_version='wuji_beta_local_z_left_p90_right_m90_v1',model='Wuji hand2_beta early v2026.6.27',
                     pose_target='Wuji l_base_link / r_base_link; not FR3 flange',side_order=['left','right'])
        for i,s in enumerate(['left','right']):
            cfg['sides'][s]=dict(R_board_center_hand=np.eye(3).tolist() if i==0 else [[0,-1,0],[1,0,0],[0,0,1]])
            (c/'calibration'/(s+'.json')).write_text(json.dumps(dict(T_eef_board=np.eye(4).tolist())))
            poses=np.tile(np.eye(4),(2,1,1));poses[:,:3,3]=[[.5,.4-i*.8,.1],[.51,.4-i*.8,.1]]
            payload.update({s+'_root_pose':poses,s+'_position_m':poses[:,:3,3],s+'_quaternion_wxyz':np.tile([1,0,0,0],(2,1)),s+'_joint_names':model_names(s),s+'_joint_positions_rad':np.zeros((2,20))})
        (c/'alignment_config.json').write_text(json.dumps(cfg))
        limits=root/'limits.json';limits.write_text(json.dumps({s:[[-2,2]]*20 for s in ['left','right']}))
        source=root/'input.npz';np.savez(source,**payload)
        return source,c,limits,payload

    def test_source_frame_preserved_and_rigid_placement_preserves_bimanual_pose(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);source,c,limits,payload=self.fixture(root);scene=root/'scene.npz'
            convert(source,scene,c,limits)
            with np.load(scene) as a:
                self.assertEqual(json.loads(str(a['metadata_json']))['frame'],'lychee_root')
                np.testing.assert_array_equal(a['hand_root_T_source_scene'][:,0],payload['left_root_pose'])
                np.testing.assert_array_equal(a['hands'],0)
            g=np.eye(4);g[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];g[:3,3]=[.2,.1,-.3]
            placed=root/'placed.npz';m=apply(scene,placed,g,'test')
            self.assertEqual(m['frame'],'tron2_base_link')
            self.assertLess(m['task_placement']['bimanual_relative_matrix_error_max'],1e-12)
            with self.assertRaises(ValueError):apply(placed,root/'twice.npz',g,'test')

    def test_disagreeing_quaternion_is_rejected(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);source,c,limits,p=self.fixture(root)
            p['right_quaternion_wxyz']=np.tile([0,0,0,1],(2,1));np.savez(source,**p)
            with self.assertRaisesRegex(ValueError,'Quaternion/matrix'):convert(source,root/'out.npz',c,limits)


if __name__=='__main__':unittest.main()

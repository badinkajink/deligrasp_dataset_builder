from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pandas as pd

class DeligraspDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(16,),
                            dtype=np.float64,
                            doc='Robot state, consists of [6x robot joint angles, '
                                '6x end-effector position (x,y,z,rx,ry,rz relative to base frame), '
                                '1x gripper position, 1x gripper applied force, ' 
                                '1x gripper contact force, 1x action_blocked flag].',
                        ),
                        'cartesian_position': tfds.features.Tensor( # this should be ee_pose but I'm following DROID
                            shape=(6,),
                            dtype=np.float64,
                            doc='6x end-effector pose (x,y,z,rx,ry,rz relative to base frame)',
                        ),
                        'joint_position': tfds.features.Tensor( # this should be ee_pose but I'm following DROID
                            shape=(6,),
                            dtype=np.float64,
                            doc='UR5 6DoF joint positions (q0, q1, q2, q3, q4, q5)',
                        ),                      
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='gripper position',
                        ),
                        'applied_force': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='gripper applied force',
                        ),
                        'contact_force': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='gripper measured contact force',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(9,),
                        dtype=np.float64,
                        doc='Robot action, consists of delta values across [6x ee pos (x, y, z, r, p, y), '
                            '1x delta gripper position, 1x delta gripper applied force, 1x terminate episode].',
                    ),
                    'action_dict': tfds.features.FeaturesDict({
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='end effector pose delta, relative to base frame',
                        ),
                        'translation': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float64,
                            doc='end effector translation delta, relative to base frame',
                        ),
                        'rotation': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float64,
                            doc='end effector rotation delta, relative to base frame',
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='gripper position delta.',
                        ),
                        'gripper_force': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='gripper force delta.',
                        ),
                    }),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'timestep_pad_mask': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='False on first step of the episode if context window==2 and for padded steps'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'subtask': tfds.features.Text(
                        doc='Language Instruction for subtask.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),                
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode_*.npy'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)  # WERE DOING IT LIVE
            columns = ['timestamp', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'aperture', 'd_aperture', 'applied_force', 'd_applied_force', 'contact_force', 'subtask', 'task', 'img', 'wrist_img']
            df = pd.DataFrame(data, columns=columns)
            # # only use first two rows, debugging
            # df = df.head(1)
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            ctr = 0
            for i, step in df.iterrows():
                # compute Kona language embedding
                language_embedding = self._embed([step['task']])[0].numpy()
                state = np.array([step['q0'], step['q1'], step['q2'], step['q3'], step['q4'], step['q5'], 
                                  step['x'], step['y'], step['z'], step['rx'], step['ry'], step['rz'], 
                                  step['aperture'], step['applied_force'], step['contact_force'], False]) # action_blocked is always False
                action = np.array([step['dx'], step['dy'], step['dz'], step['drx'], step['dry'], step['drz'],
                                   step['d_aperture'], step['d_applied_force'], i == (len(data) - 1)]) # terminate episode is 1 on last step
                grippper_position = np.array([step['aperture']])
                episode.append({
                    'observation': {
                        'image': step['img'],
                        'wrist_image': step['wrist_img'],
                        'state': state,
                        'cartesian_position': np.array([step['x'], step['y'], step['z'], step['rx'], step['ry'], step['rz']]),
                        'joint_position': np.array([step['q0'], step['q1'], step['q2'], step['q3'], step['q4'], step['q5']]),
                        'gripper_position': np.array([step['aperture']]),
                        'applied_force': np.array([step['applied_force']]),
                        'contact_force': np.array([step['contact_force']]),
                    },
                    'action': action,
                    'action_dict': {
                        'cartesian_position': np.array([step['dx'], step['dy'], step['dz'], step['drx'], step['dry'], step['drz']]),
                        'translation': np.array([step['dx'], step['dy'], step['dz']]),
                        'rotation': np.array([step['drx'], step['dry'], step['drz']]),
                        'gripper_position': np.array([step['d_aperture']]),
                        'gripper_force': np.array([step['d_applied_force']]),
                    },
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'timestep_pad_mask': ctr > 0,  # False on first step if context window==2
                    'language_instruction': step['task'],
                    'subtask': step['subtask'],
                    'language_embedding': language_embedding,
                })
                ctr += 1
                # break
            # if ctr < 50, we're going to pad the episode with duplicates of the last step, with all d_ terms zeroed out
            last_step = episode[-1].copy()
            last_step['action'] = np.zeros_like(last_step['action'])
            last_step['action_dict']['cartesian_position'] = np.zeros_like(last_step['action_dict']['cartesian_position'])
            last_step['action_dict']['translation'] = np.zeros_like(last_step['action_dict']['translation'])
            last_step['action_dict']['rotation'] = np.zeros_like(last_step['action_dict']['rotation'])
            last_step['action_dict']['gripper_position'] = np.array([0.0])
            last_step['action_dict']['gripper_force'] = np.array([0.0])
            pad_len = 50 - ctr
            # for _ in range(pad_len):
            #     episode.append(last_step.copy())            
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)
            # break
        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )


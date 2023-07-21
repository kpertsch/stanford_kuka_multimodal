from typing import Iterator, Tuple, Any

import glob
import numpy as np
import h5py
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


def dictlist2listdict(DL):
    " Converts a dict of lists to a list of dicts "
    return [dict(zip(DL,t)) for t in zip(*DL.values())]


class StanfordKukaMultimodalDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    DEFAULT_LANGUAGE_INSTRUCTION = 'insert the peg into the hole'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        self._instruction_embedding = self._embed([self.DEFAULT_LANGUAGE_INSTRUCTION])[0].numpy()

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'depth_image': tfds.features.Tensor(
                            shape=(128, 128, 1),
                            dtype=np.float32,
                            doc='Main depth camera observation.',
                        ),
                        'optical_flow': tfds.features.Tensor(
                            shape=(128, 128, 2),
                            dtype=np.float32,
                            doc='Optical flow.',
                        ),
                        'contact': tfds.features.Tensor(
                            shape=(50,),
                            dtype=np.float32,
                            doc='Robot contact information.',
                        ),
                        'ee_forces_continuous': tfds.features.Tensor(
                            shape=(50, 6),
                            dtype=np.float32,
                            doc='Robot end-effector forces.',
                        ),
                        'ee_orientation': tfds.features.Tensor(
                            shape=(4,),
                            dtype=np.float32,
                            doc='Robot end-effector orientation quaternion.',
                        ),
                        'ee_position': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='Robot end-effector position.',
                        ),
                        'ee_orientation_vel': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='Robot end-effector orientation velocity.',
                        ),
                        'ee_vel': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='Robot end-effector velocity.',
                        ),
                        'ee_yaw': tfds.features.Tensor(
                            shape=(4,),
                            dtype=np.float32,
                            doc='Robot end-effector yaw.',
                        ),
                        'ee_yaw_delta': tfds.features.Tensor(
                            shape=(4,),
                            dtype=np.float32,
                            doc='Robot end-effector yaw delta.',
                        ),
                        'joint_pos': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint positions.',
                        ),
                        'joint_vel': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint velocities.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot proprioceptive information, [7x joint pos, 1x gripper open/close].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(4,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x EEF position, '
                            '1x gripper open/close].',
                    ),
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
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/Users/karl/Downloads/triangle_real_data/*.h5'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            with h5py.File(episode_path, 'r') as F:
                data = {
                    'image': list(F['image'][()]),
                    'depth_image': list(F['depth_data'][()]),
                    'optical_flow': list(F['optical_flow'][()]),
                    'contact': list(F['contact'][()]),
                    'ee_forces_continuous': list(F['ee_forces_continuous'][()]),
                    'ee_ori': list(F['ee_ori'][()]),
                    'ee_pos': list(F['ee_pos'][()]),
                    'ee_vel': list(F['ee_vel'][()]),
                    'ee_vel_ori': list(F['ee_vel_ori'][()]),
                    'ee_yaw': list(F['ee_yaw'][()]),
                    'ee_yaw_delta': list(F['ee_yaw_delta'][()]),
                    'joint_pos': list(F['joint_pos'][()]),
                    'joint_vel': list(F['joint_vel'][()]),
                    'proprio': list(F['proprio'][()]),
                    'action': list(F['action'][()]),
                }
            data = dictlist2listdict(data)

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                episode.append({
                    'observation': {
                        'image': step['image'],
                        'depth_image': step['depth_image'],
                        'contact': step['contact'].astype(np.float32),
                        'ee_forces_continuous': step['ee_forces_continuous'].astype(np.float32),
                        'optical_flow': step['optical_flow'],
                        'ee_orientation': step['ee_ori'].astype(np.float32),
                        'ee_position': step['ee_pos'].astype(np.float32),
                        'ee_orientation_vel': step['ee_vel_ori'].astype(np.float32),
                        'ee_vel': step['ee_vel'].astype(np.float32),
                        'ee_yaw': step['ee_yaw'].astype(np.float32),
                        'ee_yaw_delta': step['ee_yaw_delta'].astype(np.float32),
                        'joint_pos': step['joint_pos'].astype(np.float32),
                        'joint_vel': step['joint_vel'].astype(np.float32),
                        'state': step['proprio'].astype(np.float32),
                    },
                    'action': step['action'].astype(np.float32),
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': self.DEFAULT_LANGUAGE_INSTRUCTION,
                    'language_embedding': self._instruction_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {}
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )


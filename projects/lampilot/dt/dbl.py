import json
import numpy as np


class DbLv1Dataset:
    def __init__(self,
                 config_root: str,
                 shuffle: bool = False,
                 seed: int = 0,
                 ):
        self.config_root = config_root
        config_list = open(f"{config_root}/config_list.txt", 'r').read().split('\n')
        self.configs = []
        for config_file_name in config_list:
            config_file_path = f"{config_root}/{config_file_name}"
            config = json.load(open(config_file_path, 'r'))
            self.configs.append({
                'name': config_file_name,
                'samples': config['samples'],
                'commands': config['commands'],
            })

        self.data_items = []
        for config in self.configs:
            for sample_idx, sample in enumerate(config['samples']):
                for command_idx, command in enumerate(config['commands']):
                    self.data_items.append({
                        'id': f"{config['name'].split('.')[0]}_s{sample_idx}_c{command_idx}",
                        'sample': sample,
                        'command': command,
                    })

        if shuffle:
            np.random.seed(seed)
            self.data_items = np.random.permutation(self.data_items)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.data_items[idx]

    def _command_stat(self):
        command_stat = []
        for config in self.configs:
            for command in config['commands']:
                command_stat.extend([len(command.split())] * len(config['samples']))

        return {
            'mean': np.mean(command_stat),
            'max': np.max(command_stat),
            'min': np.min(command_stat),
        }


class DbLv1DemoDataset(DbLv1Dataset):
    demos = [
        "dec_abs_dis25_s1_c0",
        "dec_abs_speed10_s0_c0",
        "dec_rel_dis15_s1_c0",
        "dec_rel_speed6_s0_c0",
        "go_straight_s1_c0",
        "left_lc_s0_c0",
        "right_lc_s0_c0",
        "left_overtake_s0_c0",
        "right_overtake_s0_c0",
        "turn_left_s1_c0",
        "turn_right_s1_c0",
        "pull_over_s0_c5"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_items = [item for item in self.data_items if item['id'] in self.demos]

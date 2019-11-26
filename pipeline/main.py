import argparse

from os.path import isfile, exists, join
from shutil import rmtree
from os import mkdir, remove

from features import get_feature_build_func


class StageManager(object):

    stages = {
        # feature stages
        'f-bands': get_feature_build_func('bands'),
        'f-coh-alpha': get_feature_build_func('coh-alpha'),
        'f-coh-beta': get_feature_build_func('coh-beta'),
        'f-coh-theta': get_feature_build_func('coh-theta'),
        'f-env-alpha': get_feature_build_func('env-alpha'),
        'f-env-beta': get_feature_build_func('env-beta'),
        'f-env-theta': get_feature_build_func('env-theta'),
        'f-env': get_feature_build_func('env'),
        'f-coh': get_feature_build_func('coh'),
        # model stages
    }

    def __init__(self, data_path, out_path):
        self.data_path = data_path
        self.out_path = out_path

    def run(self, stage):
        if stage not in self.stages:
            raise Exception('No such stage')

        self.stages[stage](self.data_path, self.out_path)

    def get_stages(self):
        return list(self.stages.keys())

    def check_stages(self, stages=[]):
        if stages is None:
            return
        for stage in stages:
            if stage not in self.stages:
                raise Exception('Wrong stage name - ', stage)

    def iter_stages(self, stages=[]):
        if stages is None:
            stages = self.get_stages()

        for stage in stages:
            yield self.stages[stage]


def run_grid(data_path, out_path, stages=[], clear=False):

    if not exists(out_path):
        raise Exception('Out path does not exist')

    if not exists(data_path):
        raise Exception('Data path does not exist')

    if clear:
        rmtree(out_path)
        mkdir(out_path)

    stage_manager = StageManager(data_path, out_path)

    stage_manager.check_stages(stages)

    for stage_func in stage_manager.iter_stages(stages):
        stage_func(data_path, out_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Running script')

    parser.add_argument('--data-path', action='store', type=str, required=True,
                        help='')

    parser.add_argument('--out-path', action='store', type=str, required=True,
                        help='')

    parser.add_argument('--clear', action='store_true')

    parser.add_argument('--stages', action='append')


    args = parser.parse_args()

    print('Started grid pipeline')
    # if specified take CLI param else look at config

    run_grid(args.data_path, args.out_path, stages=args.stages, clear=args.clear)

# Usage
# Filter interactive data: python scripts/filter_interactive_data.py -i DATA_DIR -o INTERACTIVE_DATA_DIR
# Filter interactive data with a specific type (i.e. v2v): python scripts/filter_interactive_data.py -i DATA_DIR -o INTERACTIVE_DATA_DIR --type v2v

# Filter typed data (i.e. p): python scripts/filter_interactive_data.py -i DATA_DIR -o TYPE_DATA_DIR --filter-marginal-type --marginal-type p

import argparse
import os

import numpy as np
import tensorflow as tf
import tqdm

import pickle5 as pickle

# Example field definition
roadgraph_features = {
    'roadgraph_samples/dir':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    'roadgraph_samples/id':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/type':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/valid':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/xyz':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)
features_description['scenario/id'] = tf.io.FixedLenFeature([1], tf.string, default_value=None)
features_description['state/objects_of_interest'] = tf.io.FixedLenFeature([128], tf.int64, default_value=None)

# vehicle = 1, pedestrian = 2, cyclist = 3
OBJECT_TYPE_DICT = {"v2v": [1, 1], "v2p": [1, 2], "v2c": [1, 3]}
TYPE_DICT = {"v": 1, "p": 2, "c": 3}


def extract_interaction_examples(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    interactive_type = args.type
    fnames = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    total_sample_cnt = 0
    total_interactive_cnt = 0
    total_interactive_type_cnt = 0
    total_type_cnt = {'v': 0, 'p': 0, 'c': 0, 'o': 0}

    if args.count_relation:
        assert args.relation_path is not None, "Need to provide --relation-path"
        with open(args.relation_path, 'rb') as f:
            relation_data = pickle.load(f)

        relation_by_types = {(1, 1): 0, (1, 2): 0, (1, 3): 0,
                             (2, 1): 0, (2, 2): 0, (2, 3): 0,
                             (3, 1): 0, (3, 2): 0, (3, 3): 0}

    for fname in tqdm.tqdm(fnames):
        dataset = tf.data.TFRecordDataset(os.path.join(input_dir, fname), compression_type='')
        interactive_data = []
        sample_cnt = 0
        interactive_cnt = 0
        interactive_type_cnt = 0

        for data in dataset:
            sample_cnt += 1

            parsed = tf.io.parse_single_example(data, features_description)
            objects_indices = parsed['state/objects_of_interest'].numpy() > 0
            objects_id = parsed['state/id'].numpy()[objects_indices]
            objects_type = parsed['state/type'].numpy()[objects_indices]
            scenario_id = parsed['scenario/id'].numpy()[0]

            if args.skip_filtering:
                if args.count_relation:
                    if scenario_id.decode('utf-8') in relation_data:
                        relation_ids = relation_data[scenario_id.decode('utf-8')][:2].tolist()
                    elif scenario_id in relation_data:
                        relation_ids = relation_data[scenario_id][:2].tolist()
                    else:
                        print("scenario: {} not in relation pickle file".format(scenario_id))
                        continue

                    # Flip object types so that the first object is always influencer.
                    if relation_ids != objects_id.tolist():
                        objects_type = np.flip(objects_type)

                    relation_by_types[tuple(objects_type.tolist())] += 1
                continue
            if args.count_type:
                for object_type in objects_type:
                    if object_type == 1:
                        total_type_cnt['v'] += 1
                    elif object_type == 2:
                        total_type_cnt['p'] += 1
                    elif object_type == 3:
                        total_type_cnt['c'] += 1
                    else:
                        total_type_cnt['o'] += 1

            if np.sum(objects_indices) == 2:
                interactive_cnt += 1
                objects_type.sort()
                objects_type = objects_type.tolist()

                if interactive_type is not None:
                    if interactive_type == 'v2v' and objects_type == OBJECT_TYPE_DICT["v2v"]:
                        interactive_data.append(data)
                        interactive_type_cnt += 1
                    elif interactive_type == 'v2p' and objects_type == OBJECT_TYPE_DICT["v2p"]:
                        interactive_data.append(data)
                        interactive_type_cnt += 1
                    elif interactive_type == 'v2c' and objects_type == OBJECT_TYPE_DICT["v2c"]:
                        interactive_data.append(data)
                        interactive_type_cnt += 1
                    elif interactive_type == 'others' and objects_type not in [OBJECT_TYPE_DICT["v2v"], OBJECT_TYPE_DICT["v2p"], OBJECT_TYPE_DICT["v2c"]]:
                        interactive_data.append(data)
                        interactive_type_cnt += 1
                else:
                    interactive_data.append(data)
            else:
                continue

        # Write interactive data.
        if len(interactive_data) > 0:
            with tf.io.TFRecordWriter(os.path.join(output_dir, fname)) as writer:
                for data in interactive_data:
                    writer.write(data.numpy())

        if interactive_type is not None:
            print('Saving {}/{} files.'.format(interactive_type_cnt, sample_cnt))
        else:
            print('Saving {}/{} files.'.format(interactive_cnt, sample_cnt))

        total_sample_cnt += sample_cnt
        total_interactive_cnt += interactive_cnt
        total_interactive_type_cnt += interactive_type_cnt

    print('Total counts {}, interactive {}, interactive type {}'.format(
        total_sample_cnt, total_interactive_cnt, total_interactive_type_cnt)
    )
    if args.count_type:
        print('Type count:', total_type_cnt)
    if args.count_relation:
        print('Relation count (inf, rea)', relation_by_types)


def extract_type_examples(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    interactive_type = args.type
    fnames = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    total_sample_cnt = 0
    total_type_cnt = 0
    agent_type_to_filter = TYPE_DICT[args.marginal_type]

    for fname in tqdm.tqdm(fnames):
        dataset = tf.data.TFRecordDataset(os.path.join(input_dir, fname), compression_type='')
        type_data = []
        sample_cnt = 0
        type_cnt = 0

        for data in dataset:
            sample_cnt += 1

            parsed = tf.io.parse_single_example(data, features_description)
            objects_indices = parsed['state/tracks_to_predict'].numpy() > 0
            objects_type = parsed['state/type'].numpy()[objects_indices]

            if agent_type_to_filter in objects_type:
                type_data.append(data)
                type_cnt += 1
            else:
                continue

        # Write interactive data.
        if len(type_data) > 0:
            with tf.io.TFRecordWriter(os.path.join(output_dir, fname)) as writer:
                for data in type_data:
                    writer.write(data.numpy())

        print('Saving {}/{} files.'.format(type_cnt, sample_cnt))

        total_sample_cnt += sample_cnt
        total_type_cnt += type_cnt

    print('Total counts {}, {} type {}'.format(total_sample_cnt, args.marginal_type, total_type_cnt))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input-dir', help='Input tfrecord files to read')
    parser.add_argument('-o', '--output-dir', help='Output tf record files to save')
    parser.add_argument('--skip-filtering', action="store_true", help='Skip filtering and only count the number of samples.')
    parser.add_argument('--type', type=str, help='Interative agent type (v2v, v2p, v2c, others).')
    parser.add_argument('--count-type', action="store_true", help='Count agent types in interactive pairs.')

    parser.add_argument('--filter-marginal-type', action="store_true", help='Filter data by marginal type.')
    parser.add_argument('--marginal-type', type=str, default="v", choices=["v", "p", "c"], help='Marginal agent type to filter.')

    parser.add_argument('--count-relation', action="store_true", help='Count reactor influncer relations.')
    parser.add_argument('--relation-path', type=str, help='Path of relation file.')
    args = parser.parse_args()

    if args.filter_marginal_type:
        extract_type_examples(args)
    else:
        extract_interaction_examples(args)


if __name__ == '__main__':
    main()

import argparse
import itertools
import logging
import os
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm as tqdm_

import pickle
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def compile_pyx_files():
    if True:
        os.chdir('src/')
        if not os.path.exists('utils_cython.c') or not os.path.exists('utils_cython.cpython-36m-x86_64-linux-gnu.so') or \
                os.path.getmtime('utils_cython.pyx') > os.path.getmtime('utils_cython.cpython-36m-x86_64-linux-gnu.so'):
            os.system('cython -a utils_cython.pyx && python setup.py build_ext --inplace')
        os.chdir('../')


# Comment out this line if pyx files have been compiled manually.
compile_pyx_files()

import utils, structs, globals
from modeling.vectornet import VectorNet

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
tqdm = partial(tqdm_, dynamic_ncols=True)


def is_main_device(device):
    return isinstance(device, torch.device) or device == 0


def learning_rate_decay(args, i_epoch, optimizer, optimizer_2=None):
    utils.i_epoch = i_epoch

    if i_epoch > 0 and i_epoch % 5 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= args.weight_decay


def gather_and_output_motion_metrics(args, device, queue, motion_metrics, metric_names, MotionMetrics):
    if is_main_device(device):
        for i in range(args.distributed_training - 1):
            motion_metrics_ = queue.get()
            assert isinstance(motion_metrics_, MotionMetrics), type(motion_metrics_)
            motion_metrics_.args = args
            for each in zip(*motion_metrics_.get_all()):
                motion_metrics.update_state(*each)
        print('all metric_values', len(motion_metrics.get_all()[0]))

        score_file = utils.get_eval_identifier()

        utils.logging(utils.metric_values_to_string(motion_metrics.result(), metric_names),
                      type=score_file, to_screen=True, append_time=True)

    else:
        queue.put(motion_metrics)


def gather_and_output_others(args, device, queue, motion_metrics):
    if is_main_device(device):
        for i in range(args.distributed_training - 1):
            other_errors_dict_ = queue.get()
            for key in utils.other_errors_dict:
                utils.other_errors_dict[key].extend(other_errors_dict_[key])

        score_file = score_file = utils.get_eval_identifier()
        utils.logging('other_errors {}'.format(utils.other_errors_to_string()),
                      type=score_file, to_screen=True, append_time=True)
        if 'train_relation' in args.other_params:
            # save relationship results
            structs.save(globals.sun_1_pred_relations, args.output_dir, utils.get_eval_identifier(),
                         prefix='pred_relations')


    else:
        queue.put(utils.other_errors_dict)


def single2joint(pred_trajectory, pred_score, args):
    assert pred_trajectory.shape == (2, 6, args.future_frame_num, 2)
    assert np.all(pred_score < 0)
    pred_score = np.exp(pred_score)
    li = []
    scores = []
    for i in range(6):
        for j in range(6):
            score = pred_score[0, i] * pred_score[0, j]
            scores.append(score)
            li.append((score, i, j))

    argsort = np.argsort(-np.array(scores))

    pred_trajectory_joint = np.zeros((6, 2, args.future_frame_num, 2))
    pred_score_joint = np.zeros(6)

    for k in range(6):
        score, i, j = li[argsort[k]]
        pred_trajectory_joint[k, 0], pred_trajectory_joint[k, 1] = pred_trajectory[0, i], pred_trajectory[1, j]
        pred_score_joint[k] = score

    return np.array(pred_trajectory_joint), np.array(pred_score_joint)


def pair2joint(pred_trajectory, pred_score, args):
    assert pred_trajectory.shape == (2, 6, args.future_frame_num, 2)

    pred_trajectory_joint = np.zeros((6, 2, args.future_frame_num, 2))
    pred_score_joint = np.zeros(6)
    for k in range(6):
        assert utils.equal(pred_score[0, k], pred_score[1, k])
        pred_trajectory_joint[k, 0] = pred_trajectory[0, k]
        pred_trajectory_joint[k, 1] = pred_trajectory[1, k]
        pred_score_joint[k] = pred_score[0, k]

    return pred_trajectory_joint, pred_score_joint


def get_metric_params(mapping, args):
    gt_trajectory = tf.convert_to_tensor(utils.get_from_mapping(mapping, 'gt_trajectory'))
    gt_is_valid = tf.convert_to_tensor(utils.get_from_mapping(mapping, 'gt_is_valid'))
    object_type = tf.convert_to_tensor(utils.get_from_mapping(mapping, 'object_type'))
    scenario_id = utils.get_from_mapping(mapping, 'scenario_id')[0]
    object_id = tf.convert_to_tensor(utils.get_from_mapping(mapping, 'object_id'))

    if 'joint_eval' in args.other_params:
        assert len(mapping) == 2, len(mapping)
        gt_trajectory = gt_trajectory[tf.newaxis, :]
        gt_is_valid = gt_is_valid[tf.newaxis, :]
        object_type = object_type[tf.newaxis, :]

    return (
        gt_trajectory,
        gt_is_valid,
        object_type,
        scenario_id,
        object_id
    )


eval_rst_to_save = {}
total_yaw_loss = 0


def eval_one_epoch(model, iter_bar, optimizer, device, args: utils.Args, i_epoch,
                   queue=None, rst_save_queue=None, optimizer_2=None):
    global total_yaw_loss

    utils.other_errors_dict.clear()
    start_time = time.time()

    length = len(iter_bar)
    if not args.debug_mode:
        assert dist.get_world_size() == args.distributed_training

    model.eval()
    logger.info("***** Recover model: %s *****", args.model_recover_path)
    if args.model_recover_path is None:
        raise ValueError("model_recover_path not specified.")
    model_recover = torch.load(args.model_recover_path)
    model.module.load_state_dict(model_recover)

    import tensorflow as tf
    global tf
    from waymo_tutorial import MotionMetrics, metrics_config, metric_names

    motion_metrics = MotionMetrics(metrics_config)

    if args.mode_num != 6:
        motion_metrics.not_compute = True

    utils.motion_metrics = MotionMetrics(metrics_config)
    utils.metric_names = metric_names

    if 'save_rst' in args.other_params:
        eval_rst_to_save_this_epoch = {}

    for step, batch in enumerate(iter_bar):
        if args.waymo:
            if isinstance(iter_bar, tqdm_):
                dataset = iter_bar.iterable
            else:
                dataset = iter_bar

            if batch is None:
                # dist.barrier()
                sufficient, length = dataset.waymo_generate()
                if not sufficient and length == 0:
                    break
                else:
                    continue

        assert batch is not None

        # Changes: allow evaluate only on part of the agents
        # assert args.agent_type is not None or args.inter_agent_types is not None

        metric_params = get_metric_params(batch, args)

        assert not ('interactive-single_traj' in args.other_params), 'deprecated'

        if 'pred_yaw' in args.other_params:
            # pred_trajectory: [n, k, 80, 2], pred_yaw: [n, 1, k, 80]
            pred_trajectory, pred_score, pred_yaw, _ = model(batch, device)
            # pred_yaw: [n, k, 80]
            pred_yaw = np.squeeze(pred_yaw, axis=(1,))
        else:
            pred_trajectory, pred_score, _ = model(batch, device)

        if 'save_rst' in args.other_params:
            assert args.eval_exp_path is not None, 'give export path in eval_exp_path'
            agent_ids = []
            scenario_id = None
            for each_agent in batch:
                agent_ids.append(int(each_agent['object_id']))
                scenario_id = each_agent['scenario_id']
            assert scenario_id is not None
            if 'train_reactor' in args.other_params:
                eval_rst_to_save_this_epoch[scenario_id] = {
                    'rst': pred_trajectory[0],
                    'score': pred_score[0],
                    'ids': agent_ids[0]
                }
            else:
                eval_rst_to_save_this_epoch[scenario_id] = {
                    'rst': pred_trajectory,
                    'score': pred_score,
                    'ids': agent_ids
                }
            eval_rst_to_save.update(eval_rst_to_save_this_epoch)

        if 'single2joint' in args.other_params:
            pred_trajectory, pred_score = single2joint(pred_trajectory, pred_score, args)
        elif 'tnt_square' in args.other_params:
            pred_trajectory, pred_score = pair2joint(pred_trajectory, pred_score, args)

        # if pred_score is None:
        #     pred_score = np.ones(pred_trajectory.shape[:-2], dtype=np.float32)
        if 'joint_eval' in args.other_params:
            assert pred_trajectory.shape == (6, 2, args.future_frame_num, 2)
            pred_trajectory = pred_trajectory[np.newaxis, :]
            pred_score = pred_score[np.newaxis, :]

        if 'train_relation' not in args.other_params:
            motion_metrics.args = args
            motion_metrics.update_state(
                tf.convert_to_tensor(pred_trajectory[:, 0].astype(np.float32)),
                tf.convert_to_tensor(pred_score.astype(np.float32)),
                *metric_params
            )

    if 'save_rst' in args.other_params:
        if is_main_device(device):
            merged_eval_rst = eval_rst_to_save
            for i in range(args.distributed_training - 1):
                eval_rst_to_save_ = rst_save_queue.get()
                assert isinstance(eval_rst_to_save_, type({})), type(eval_rst_to_save_)
                merged_eval_rst.update(eval_rst_to_save_)
            number_str = args.eval_rst_saving_number
            if number_str is None:
                file_path_1 = args.eval_exp_path+'.pickle'
            else:
                file_path_1 = args.eval_exp_path + number_str + '.pickle'
            file_path_2 = file_path_1
            print('saving merged result with ', len(list(merged_eval_rst.keys())), ' scenarios', ' at \n', file_path_1, '\n' ,file_path_2)
            with open(file_path_1, 'wb') as f:
                pickle.dump(merged_eval_rst, f, pickle.HIGHEST_PROTOCOL)
                structs.save(obj=merged_eval_rst, output_dir=args.output_dir,
                             eval_identifier=file_path_2)
        else:
            rst_save_queue.put(eval_rst_to_save)

    gather_and_output_motion_metrics(args, device, queue, motion_metrics, metric_names, MotionMetrics)

    dist.barrier()

    gather_and_output_others(args, device, queue, motion_metrics)


def train_one_epoch(model, iter_bar, optimizer, device, args: utils.Args, i_epoch, queue=None, optimizer_2=None):
    li_ADE = []
    li_FDE = []
    utils.other_errors_dict.clear()
    start_time = time.time()
    if args.distributed_training:
        assert dist.get_world_size() == args.distributed_training

    if args.train_extra and i_epoch == 0:
        logger.info("***** Recover model: %s *****", args.model_recover_path)
        if args.model_recover_path is None:
            raise ValueError("model_recover_path not specified.")
        model_recover = torch.load(args.model_recover_path)
        if 'loading_partial_weights' in args.other_params:
            # update model for missing keys
            model_dict = model.module.state_dict()
            loaded_state_dict = {k:v for k,v in model_recover.items() if k in model_dict.keys() and 'decoder' not in k}
            model_dict.update(loaded_state_dict)
            model.module.load_state_dict(model_dict)
        else:
            model.module.load_state_dict(model_recover)

    for step, batch in enumerate(iter_bar):
        if args.waymo:
            if isinstance(iter_bar, tqdm_):
                dataset = iter_bar.iterable
            else:
                dataset = iter_bar

            if batch is None:
                if is_main_device(device):
                    iter_bar.set_description('loading data...')
                sufficient, length = dataset.waymo_generate()
                if args.distributed_training:
                    queue.put(length)

                    if is_main_device(device):
                        lengths = []
                        for i in range(args.distributed_training):
                            lengths.append(queue.get())
                        assert queue.empty()
                        length = np.min(lengths)
                        for i in range(args.distributed_training):
                            queue.put(length)

                    dist.barrier()
                    length = queue.get()
                    if length == 0:
                        break
                    else:
                        dataset.set_ex_list_length(length)

                continue

        if 'train_pair_interest' in args.other_params:
            batch = [batch[i][j] for i, j in itertools.product(range(len(batch)), range(2))]
            loss, DE, _ = model(batch, device)
            loss.backward()
        else:
            loss, DE, _ = model(batch, device)
            loss.backward()

        if is_main_device(device):
            iter_bar.set_description(f'loss={loss.item():.3f}')

        final_idx = batch[0].get('final_idx', -1)
        li_FDE.extend([each for each in DE[:, final_idx]])

        if optimizer_2 is not None:
            optimizer_2.step()
            optimizer_2.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

    if not args.debug and is_main_device(device):
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(
            args.model_save_dir, "model.{0}.bin".format(i_epoch + 1))
        torch.save(model_to_save.state_dict(), output_model_file)

    if args.waymo:
        if is_main_device(device):
            print()
            miss_rates = (utils.get_miss_rate(li_FDE, dis=2.0), utils.get_miss_rate(li_FDE, dis=4.0),
                          utils.get_miss_rate(li_FDE, dis=6.0))

            utils.logging(f'ADE: {np.mean(li_ADE) if len(li_ADE) > 0 else None}',
                          f'FDE: {np.mean(li_FDE) if len(li_FDE) > 0 else None}',
                          f'MR(2,4,6): {miss_rates}',
                          utils.other_errors_to_string(),
                          type='train_loss', to_screen=True)

            if 'loading_sum' in args.other_params:
                print(f"summary: {dataset.loading_summary}")


def demo_basic(rank, world_size, kwargs, queue, queue2):
    args = kwargs['args']
    if args.waymo:
        import tensorflow as tf

    if world_size > 0:
        print(f"Running DDP on rank {rank}.")

        def setup(rank, world_size):
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = args.master_port

            # initialize the process group
            dist.init_process_group("nccl", rank=rank, world_size=world_size)

        setup(rank, world_size)

        utils.args = args
        model = VectorNet(args).to(rank)

        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model = VectorNet(args).to(rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if rank == 0 and world_size > 0:
        receive = queue.get()
        assert receive == True

    if args.distributed_training:
        dist.barrier()
    args.reuse_temp_file = True

    if args.waymo:
        from dataset_waymo import Dataset
        train_dataset = Dataset(args, args.train_batch_size, rank=rank, to_screen=False)
        train_sampler = train_dataset
        train_dataloader = train_dataset
    for i_epoch in range(int(args.num_train_epochs)):
        learning_rate_decay(args, i_epoch, optimizer)
        utils.logging(optimizer.state_dict()['param_groups'])
        if rank == 0:
            print('Epoch: {}/{}'.format(i_epoch, int(args.num_train_epochs)), end='  ')
            print('Learning Rate = %5.8f' % optimizer.state_dict()['param_groups'][0]['lr'])
        train_sampler.set_epoch(i_epoch)

        if rank == 0:
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
        else:
            iter_bar = train_dataloader

        if args.do_eval:
            if args.waymo:
                eval_one_epoch(model, iter_bar, optimizer, rank, args, i_epoch, queue, rst_save_queue=queue2)
            else:
                pass
            return
        else:
            train_one_epoch(model, iter_bar, optimizer, rank, args, i_epoch, queue)

        if args.distributed_training:
            dist.barrier()
    if args.distributed_training:
        dist.destroy_process_group()


def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("Loading dataset", args.data_dir)
    if args.waymo:
        from dataset_waymo import Dataset
    if args.distributed_training:
        queue = mp.Manager().Queue()
        # queue2 is specifically used for saving results when eval
        queue2 = mp.Manager().Queue()
        kwargs = {'args': args}
        spawn_context = mp.spawn(demo_basic,
                                 args=(args.distributed_training, kwargs, queue, queue2),
                                 nprocs=args.distributed_training,
                                 join=False)
        if args.waymo:
            pass
        else:
            train_dataset = Dataset(args, args.train_batch_size)
        queue.put(True)
        while not spawn_context.join():
            pass
    else:
        if not args.debug_mode:
            assert False, 'single gpu not supported'
        demo_basic(0, 0, {'args': args}, None, None)

        train_dataset = Dataset(args, args.train_batch_size)
        train_sampler = RandomSampler(train_dataset, replacement=False)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                                       sampler=train_sampler,
                                                       collate_fn=utils.batch_list_to_batch_tensors,
                                                       pin_memory=False)
        if args.waymo:
            train_dataloader = train_dataset

        t_total = int(len(train_dataloader) * args.num_train_epochs)

        model = VectorNet(args)
        print('torch.cuda.device_count', torch.cuda.device_count())

        if 'complete_traj-3' in args.other_params and not args.debug:
            assert False
        if args.train_extra:
            model_recover = torch.load(args.model_recover_path)
            for key, value in model_recover.items():
                utils.logging(key, type='model_recover')
            utils.load_model(model, model_recover)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)
        model.train()
        for i_epoch in range(int(args.num_train_epochs)):
            learning_rate_decay(args, i_epoch, optimizer)
            utils.logging(optimizer.state_dict()['param_groups'])
            print('Epoch: {}/{}'.format(i_epoch, int(args.num_train_epochs)), end='  ')
            print('Learning Rate = %5.8f' % optimizer.state_dict()['param_groups'][0]['lr'])
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
            train_one_epoch(model, iter_bar, optimizer, device, args, i_epoch)


def main():
    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    args: utils.Args = parser.parse_args()
    utils.init(args, logger)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    if args.waymo:
        run(args)
    else:
        assert False

    logger.info('Finish.')


if __name__ == "__main__":
    main()

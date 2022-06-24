from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import structs
import utils_cython
from modeling.lib import PointSubGraph, GlobalGraphRes, CrossAttention, GlobalGraph, MLP

import utils, globals


class DecoderRes(nn.Module):
    def __init__(self, hidden_size, out_features=60):
        super(DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class DecoderResCat(nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.hidden_size = hidden_size
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class Decoder(nn.Module):

    def __init__(self, args_: utils.Args, vectornet):
        super(Decoder, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size
        self.future_frame_num = args.future_frame_num
        self.mode_num = args.mode_num

        self.decoder = DecoderRes(hidden_size, out_features=2)

        if 'variety_loss' in args.other_params:
            self.variety_loss_decoder = DecoderResCat(hidden_size, hidden_size, out_features=6 * self.future_frame_num * 2)

            if 'variety_loss-prob' in args.other_params:
                self.variety_loss_decoder = DecoderResCat(hidden_size, hidden_size, out_features=6 * self.future_frame_num * 2 + 6)
        elif 'goals_2D' in args.other_params:
            # self.decoder = DecoderResCat(hidden_size, hidden_size, out_features=self.future_frame_num * 2)
            self.goals_2D_mlps = nn.Sequential(
                MLP(2, hidden_size),
                MLP(hidden_size),
                MLP(hidden_size)
            )
            # self.goals_2D_decoder = DecoderRes(hidden_size * 3, out_features=1)
            self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)
            self.goals_2D_cross_attention = CrossAttention(hidden_size)
            if 'point_sub_graph' in args.other_params:
                self.goals_2D_point_sub_graph = PointSubGraph(hidden_size)
            if 'raster' in args.other_params:
                if 'sub_to_decoder' in args.other_params:
                    if '2xdec' in args.other_params:
                        self.goals_2D_decoder = DecoderResCat(hidden_size * 2, hidden_size * 5, out_features=1)
                    else:
                        self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 5, out_features=1)
                else:
                    self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 4, out_features=1)
            else:
                if 'sub_to_decoder' in args.other_params:
                    self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 4, out_features=1)

            # Create subgoal decoders.
            if args.classify_sub_goals:
                self.goals_2D_decoder_3s = DecoderResCat(
                    self.goals_2D_decoder.hidden_size,
                    self.goals_2D_decoder.in_features + hidden_size * 2,
                    self.goals_2D_decoder.out_features)
                self.goals_2D_decoder_5s = DecoderResCat(
                    self.goals_2D_decoder.hidden_size,
                    self.goals_2D_decoder.in_features + hidden_size,
                    self.goals_2D_decoder.out_features)

        if 'tnt' in args.other_params or 'densetnt' in args.other_params:
            self.tnt_cross_attention = CrossAttention(hidden_size)
            if args.classify_sub_goals:
                self.tnt_decoder = DecoderResCat(hidden_size, hidden_size * 5, out_features=self.future_frame_num * 2)
            else:
                self.tnt_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=self.future_frame_num * 2)

        if 'tnt_square' in args.other_params:
            # Define input dimension based on joint target type.
            if args.joint_target_type == "pair":
                tnt_square_goals_4D_encoder_input_dim = 9
            else:
                tnt_square_goals_4D_encoder_input_dim = 5

            self.tnt_square_goals_4D_encoder = nn.Sequential(
                MLP(tnt_square_goals_4D_encoder_input_dim, hidden_size),
                MLP(hidden_size, hidden_size),
            )
            self.tnt_square_decoder = DecoderResCat(hidden_size, in_features=hidden_size * 5, out_features=1)

        if 'train_relation' in args.other_params:
            self.relation_decoder = DecoderResCat(hidden_size, in_features=hidden_size * 2, out_features=3)

        if 'direct_relation_label' in args.other_params:
            # max number of agents is 125
            self.direct_relation_decoder = DecoderResCat(hidden_size, in_features=hidden_size * 2, out_features=256)


    def goals_2D_per_example_calc_loss(self, i: int, goals_2D: np.ndarray, mapping: List[Dict], inputs: Tensor,
                                       inputs_lengths: List[int], hidden_states: Tensor, device, loss: Tensor,
                                       DE: np.ndarray, gt_points: np.ndarray, scores: Tensor, highest_goal: np.ndarray,
                                       labels_is_valid: List[np.ndarray]):
        """
        Calculate loss for a training example
        """
        final_idx = mapping[i].get('final_idx', -1)
        gt_yaws = mapping[i].get('yaw_labels')
        gt_goal = gt_points[final_idx]
        DE[i][final_idx] = np.sqrt((highest_goal[0] - gt_points[final_idx][0]) ** 2 + (highest_goal[1] - gt_points[final_idx][1]) ** 2)
        loss[i] += F.nll_loss(scores.unsqueeze(0),
                              torch.tensor([mapping[i]['goals_2D_labels']], device=device))

        if 'tnt' in args.other_params or 'densetnt' in args.other_params:
            target_feature = self.goals_2D_mlps(
                torch.tensor(goals_2D[mapping[i]['goals_2D_labels']], dtype=torch.float, device=device))
            attention_states = self.tnt_cross_attention(
                target_feature.unsqueeze(0).unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0).squeeze(0)

            if args.classify_sub_goals:
                # Classify subgoals and compute loss.
                goals_2D_tensor = torch.tensor(goals_2D, device=device, dtype=torch.float)

                target_feature_3s = self.goals_2D_mlps(
                    torch.tensor(goals_2D[mapping[i]['goals_2D_labels_3s']], dtype=torch.float, device=device))
                target_feature_5s = self.goals_2D_mlps(
                    torch.tensor(goals_2D[mapping[i]['goals_2D_labels_5s']], dtype=torch.float, device=device))
                target_feature_5s_8s = torch.cat([target_feature_5s, target_feature], 0)

                scores_3s = self.get_scores(goals_2D_tensor, inputs, hidden_states, inputs_lengths, i, mapping, device, subgoal='3s', subgoal_input=target_feature_5s_8s)
                scores_5s = self.get_scores(goals_2D_tensor, inputs, hidden_states, inputs_lengths, i, mapping, device, subgoal='5s', subgoal_input=target_feature)

                loss[i] += F.nll_loss(scores_3s.unsqueeze(0),
                                      torch.tensor([mapping[i]['goals_2D_labels_3s']], device=device))
                loss[i] += F.nll_loss(scores_5s.unsqueeze(0),
                                      torch.tensor([mapping[i]['goals_2D_labels_5s']], device=device))

                target_feature = torch.cat([target_feature_3s, target_feature_5s, target_feature], -1)
                predict_traj = self.tnt_decoder(
                    torch.cat([hidden_states[i, 0, :], target_feature, attention_states], dim=-1)).view([self.future_frame_num, 2])
                loss[i] += args.traj_loss_coeff * F.smooth_l1_loss(predict_traj, torch.tensor(gt_points, dtype=torch.float, device=device))
            else:
                predict_traj = self.tnt_decoder(
                    torch.cat([hidden_states[i, 0, :], target_feature, attention_states], dim=-1)).view([self.future_frame_num, 2])
                loss[i] += args.traj_loss_coeff * F.smooth_l1_loss(predict_traj, torch.tensor(gt_points, dtype=torch.float, device=device))

                if args.short_term_loss_coeff > 0:
                    # Add additional regression loss at critical points (3s and 5s).
                    if predict_traj.shape[0] > 30:
                        loss[i] += args.short_term_loss_coeff * F.smooth_l1_loss(predict_traj[29], torch.tensor(gt_points[29], dtype=torch.float, device=device))
                    if predict_traj.shape[0] > 50:
                        loss[i] += args.short_term_loss_coeff * F.smooth_l1_loss(predict_traj[49], torch.tensor(gt_points[49], dtype=torch.float, device=device))

    def goals_2D_per_example(self, i: int, goals_2D: np.ndarray, mapping: List[Dict], lane_states_batch: List[Tensor],
                             inputs: Tensor, inputs_lengths: List[int], hidden_states: Tensor, labels: List[np.ndarray],
                             labels_is_valid: List[np.ndarray], device, loss: Tensor, DE: np.ndarray):
        """
        :param i: example index in batch
        :param goals_2D: candidate goals sampled from map (shape ['goal num', 2])
        :param lane_states_batch: each value in list is hidden states of lanes (value shape ['lane num', hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
        :param loss: (shape [batch_size])
        :param DE: displacement error (shape [batch_size, self.future_frame_num])
        """
        if args.do_train:
            final_idx = mapping[i].get('final_idx', -1)
            assert labels_is_valid[i][final_idx]

        gt_points = labels[i].reshape([self.future_frame_num, 2])

        stage_one_topk_ids = None
        goals_2D_tensor = torch.tensor(goals_2D, device=device, dtype=torch.float)
        get_scores_inputs = (inputs, hidden_states, inputs_lengths, i, mapping, device)

        scores = self.get_scores(goals_2D_tensor, *get_scores_inputs)
        index = torch.argmax(scores).item()
        highest_goal = goals_2D[index]

        if args.do_train:
            self.goals_2D_per_example_calc_loss(i, goals_2D, mapping, inputs, inputs_lengths,
                                                hidden_states, device, loss, DE, gt_points, scores, highest_goal, labels_is_valid)

        if args.visualize:
            mapping[i]['vis.goals_2D'] = goals_2D
            mapping[i]['vis.scores'] = np.array(scores.tolist())
            mapping[i]['vis.labels'] = gt_points
            mapping[i]['vis.labels_is_valid'] = labels_is_valid[i]

        if 'tnt_square' in args.other_params:
            # _, topk_ids = torch.topk(scores, k=min(70, len(scores)))
            final_idx = mapping[i].get('final_idx', -1)
            gt_goal = gt_points[final_idx]
            assert len(goals_2D) > 0

            mapping[i]['tnt_square.pack'] = [
                *utils_cython.tnt_square_filter(goals_2D, utils.to_numpy(scores), args),
                gt_goal,
            ]

            if args.do_eval:
                if 'single2joint' in args.other_params:
                    utils.select_goals_by_NMS(mapping[i], goals_2D, np.array(scores.tolist()), args.nms_threshold, mapping[i]['speed'])

        else:
            if args.do_eval:
                if args.nms_threshold is not None:
                    utils.select_goals_by_NMS(mapping[i], goals_2D, np.array(scores.tolist()), args.nms_threshold, mapping[i]['speed'])
                else:
                    pass

    def goals_2D_eval(self, batch_size, mapping, labels, hidden_states, inputs, inputs_lengths, device):
        if 'tnt' in args.other_params or 'densetnt' in args.other_params:
            pred_goals_batch = [mapping[i]['pred_goals'] for i in range(batch_size)]
            pred_probs_batch = [mapping[i]['pred_probs'] for i in range(batch_size)]
            pred_indices_batch = [mapping[i]['pred_indices'] for i in range(batch_size)]
        else:
            pass
        pred_goals_batch = np.array(pred_goals_batch)
        pred_probs_batch = np.array(pred_probs_batch)
        assert pred_goals_batch.shape == (batch_size, self.mode_num, 2)
        assert pred_probs_batch.shape == (batch_size, self.mode_num)

        if 'tnt' in args.other_params or 'densetnt' in args.other_params:
            pred_trajs_batch = []
            for i in range(batch_size):
                targets_feature = self.goals_2D_mlps(torch.tensor(pred_goals_batch[i], dtype=torch.float, device=device))
                hidden_attention = self.tnt_cross_attention(
                    targets_feature.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0)
                if args.classify_sub_goals:
                    goals_2D = mapping[i]['goals_2D']
                    goals_2D_tensor = torch.tensor(goals_2D, device=device, dtype=torch.float)

                    # Predict subgoal for 5s.
                    goals_5s = []
                    for m in range(self.mode_num):
                        goals_5s_scores = self.get_scores(goals_2D_tensor, inputs, hidden_states, inputs_lengths, i, mapping, device, subgoal='5s', subgoal_input=targets_feature[m])
                        top_goals_5s_index = torch.argmax(goals_5s_scores)
                        goals_5s.append(goals_2D_tensor[top_goals_5s_index])
                    targets_feature_5s = self.goals_2D_mlps(torch.stack(goals_5s, 0))

                    # Predict subgoal for 3s.
                    goals_3s = []
                    for m in range(self.mode_num):
                        goals_3s_scores = self.get_scores(goals_2D_tensor, inputs, hidden_states, inputs_lengths, i, mapping, device, subgoal='3s',
                                                          subgoal_input=torch.cat([targets_feature_5s[m], targets_feature[m]]))
                        top_goals_3s_index = torch.argmax(goals_3s_scores)
                        goals_3s.append(goals_2D_tensor[top_goals_3s_index])
                    targets_feature_3s = self.goals_2D_mlps(torch.stack(goals_3s, 0))

                    targets_feature = torch.cat([targets_feature_3s, targets_feature_5s, targets_feature], -1)
                    predict_trajs = self.tnt_decoder(
                        torch.cat([hidden_states[i, 0, :].unsqueeze(0).expand(len(targets_feature), -1), targets_feature,
                                   hidden_attention], dim=-1)).view([self.mode_num, self.future_frame_num, 2])
                    predict_trajs = np.array(predict_trajs.tolist())
                else:
                    predict_trajs = self.tnt_decoder(
                        torch.cat([hidden_states[i, 0, :].unsqueeze(0).expand(len(targets_feature), -1), targets_feature,
                                   hidden_attention], dim=-1)).view([self.mode_num, self.future_frame_num, 2])
                    predict_trajs = np.array(predict_trajs.tolist())
                mapping[i]['vis.predict_trajs'] = predict_trajs.copy()
                # predict_trajs[:, -1, :] = pred_goals_batch[i]
                if args.waymo:
                    normalizer = mapping[i]['normalizer']
                    for each in predict_trajs:
                        each[:] = normalizer(each, reverse=True)
                else:
                    assert False
                pred_trajs_batch.append(predict_trajs)
                if 'pred_yaw' in args.other_params:
                    pred_yaws_batch.append(predict_yaws)
            pred_trajs_batch = np.array(pred_trajs_batch)
            if 'pred_yaw' in args.other_params:
                pred_yaws_batch = np.array(pred_yaws_batch)
        else:
            pass

        if args.visualize:
            for i in range(batch_size):
                utils.visualize_goals_2D(mapping[i], mapping[i]['vis.goals_2D'], mapping[i]['vis.scores'], self.future_frame_num,
                                         labels=mapping[i]['vis.labels'],
                                         labels_is_valid=mapping[i]['vis.labels_is_valid'],
                                         predict=mapping[i]['vis.predict_trajs'])

        if 'pred_yaw' in args.other_params:
            return pred_trajs_batch, pred_probs_batch, pred_yaws_batch, None
        else:
            return pred_trajs_batch, pred_probs_batch, None

    def variety_loss(self, mapping: List[Dict], hidden_states: Tensor, batch_size, inputs: Tensor,
                     inputs_lengths: List[int], labels_is_valid: List[np.ndarray], loss: Tensor,
                     DE: np.ndarray, device, labels: List[np.ndarray]):
        """
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param DE: displacement error (shape [batch_size, self.future_frame_num])
        """
        if 'raster' in args.other_params:
            outputs = self.variety_loss_decoder(args.raster_image_hidden)
        else:
            outputs = self.variety_loss_decoder(hidden_states[:, 0, :])
        pred_probs = None
        if 'variety_loss-prob' in args.other_params:
            pred_probs = F.log_softmax(outputs[:, -6:], dim=-1)
            outputs = outputs[:, :-6].view([batch_size, 6, self.future_frame_num, 2])
        else:
            outputs = outputs.view([batch_size, 6, self.future_frame_num, 2])

        for i in range(batch_size):
            if args.do_train:
                assert labels_is_valid[i][-1]
            gt_points = np.array(labels[i]).reshape([self.future_frame_num, 2])
            argmin = np.argmin(utils.get_dis_point_2_points(gt_points[-1], np.array(outputs[i, :, -1, :].tolist())))

            loss_ = F.smooth_l1_loss(outputs[i, argmin],
                                     torch.tensor(gt_points, device=device, dtype=torch.float), reduction='none')
            loss_ = loss_ * torch.tensor(labels_is_valid[i], device=device, dtype=torch.float).view(self.future_frame_num, 1)
            if labels_is_valid[i].sum() > utils.eps:
                loss[i] += loss_.sum() / labels_is_valid[i].sum()

            if 'variety_loss-prob' in args.other_params:
                loss[i] += F.nll_loss(pred_probs[i].unsqueeze(0), torch.tensor([argmin], device=device))
        if args.do_eval:
            outputs = np.array(outputs.tolist())
            pred_probs = np.array(pred_probs.tolist(), dtype=np.float32) if pred_probs is not None else pred_probs
            if args.waymo:
                for i in range(batch_size):
                    normalizer: utils.Normalizer = mapping[i]['normalizer']
                    for trajectory in outputs[i]:
                        trajectory[:] = normalizer(trajectory, reverse=True)

            else:
                for i in range(batch_size):
                    for each in outputs[i]:
                        utils.to_origin_coordinate(each, i)

            return outputs, pred_probs, None
        return loss.mean(), DE, None

    def forward(self, mapping: List[Dict], batch_size, lane_states_batch: List[Tensor], inputs: Tensor,
                inputs_lengths: List[int], hidden_states: Tensor, device):
        """
        :param lane_states_batch: each value in list is hidden states of lanes (value shape ['lane num', hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, 'element num', hidden_size])
        """
        labels = utils.get_from_mapping(mapping, 'labels')
        labels_is_valid = utils.get_from_mapping(mapping, 'labels_is_valid')
        loss = torch.zeros(batch_size, device=device)
        DE = np.zeros([batch_size, self.future_frame_num])

        if 'direct_relation_label' in args.other_params:
            # Compute Direct Relation Loss
            hidden = hidden_states[:, :2, :].view(batch_size, -1)
            confidences = self.direct_relation_decoder(hidden)
            label_scores = confidences.reshape(-1, 2)
            # label_scores, outputs: [b*128, 2]
            outputs = F.log_softmax(label_scores, dim=-1)
            interaction_labels = utils.get_from_mapping(mapping, 'influencer_idx')
            # loss += F.nll_loss(outputs, torch.tensor(interaction_labels, dtype=torch.long, device=device))
            loss += F.nll_loss(outputs, torch.tensor(np.array(interaction_labels).copy().flatten(), dtype=torch.long, device=device))
            argmax = torch.argmax(outputs, dim=-1)
            argmax = argmax.reshape(batch_size, 128)
            for i in range(batch_size):
                if 'pred_with_threshold' in args.other_params and args.do_eval:
                    # if confidences of both are below threshold, then predict as label 2
                    confidence_threshold = args.relation_pred_threshold
                    confidence = torch.exp(outputs[i])
                    confidence /= torch.sum(confidence)
                    if args.vehicle_r_pred_threshold is not None:
                        # check relation prediction confidence first
                        if argmax[i][0] != 1:
                            for j in range(128):
                                if confidence[argmax[i][j]] < confidence_threshold:
                                    argmax[i][j] = 0
                        # currently does not support checking confidence for vehicle as influencer and filter

                for j, label in enumerate(interaction_labels[i]):
                    ok = argmax[i][j] == interaction_labels[i][j]
                    if label == 1:
                        utils.other_errors_put('directR_inf_1', ok)
                    elif label == 0:
                        utils.other_errors_put('directR_inf_0', float(ok))
                    else:
                        assert False, f'unknown label: {label}'
                # this blows gpu memory when testing
                globals.sun_1_pred_relations[bytes.decode(mapping[i]['scenario_id'])] = [argmax[i],
                                                                                         torch.tensor(outputs[i], dtype=torch.float16)]
                # globals.sun_1_pred_relations[bytes.decode(mapping[i]['scenario_id'])] = int(argmax[i])
            if args.do_eval:
                return np.zeros([batch_size, self.future_frame_num, 2]), np.zeros(batch_size), None
            else:
                return loss.mean(), DE, None

        if 'train_relation' in args.other_params:
            # Compute Relation Loss
            hidden = hidden_states[:, :2, :].view(batch_size, -1)
            confidences = self.relation_decoder(hidden)
            outputs = F.log_softmax(confidences, dim=-1)
            interaction_labels = utils.get_from_mapping(mapping, 'interaction_label')
            loss += F.nll_loss(outputs, torch.tensor(interaction_labels, dtype=torch.long, device=device))
            argmax = torch.argmax(outputs, dim=-1)
            for i in range(batch_size):
                if 'pred_with_threshold' in args.other_params and args.do_eval:
                    # if confidences of both are below threshold, then predict as label 2
                    confidence_threshold = args.relation_pred_threshold
                    confidence = torch.exp(outputs[i])
                    confidence /= torch.sum(confidence)
                    if args.vehicle_r_pred_threshold is not None:
                        # check relation prediction confidence first
                        if confidence[0] < confidence_threshold and confidence[1] < confidence_threshold:
                            argmax[i] = 2
                        # then check confidence for vehicle as influencer
                        v_confidence_threshold = args.vehicle_r_pred_threshold
                        # when doing relation, smaller id always comes first
                        current_agent_type = mapping[i]['track_type_string'][0]
                        pair_agent_type = mapping[i]['track_type_string'][1]
                        # label = 0: smaller_id / current agent is influencer
                        # label = 1: larger_id / pair agent is influencer
                        if current_agent_type == 'vehicle' and pair_agent_type != 'vehicle':
                            # vehicle is smaller id agent, bring large threshold for vehicle as influencer
                            if confidences[0] < v_confidence_threshold:
                                argmax[i] = 2
                        elif current_agent_type != 'vehicle' and pair_agent_type == 'vehicle':
                            # vehicle is larger id agent, bring large threshold for vehicle as influencer
                            if confidences[1] < v_confidence_threshold:
                                argmax[i] = 2

                ok = argmax[i] == interaction_labels[i]
                utils.other_errors_put(f'interaction_label.{interaction_labels[i]}', float(ok))
                utils.other_errors_put('interaction_label.all', float(ok))
                # this blows gpu memory when testing
                globals.sun_1_pred_relations[bytes.decode(mapping[i]['scenario_id'])] = [int(argmax[i]), torch.tensor(outputs[i], dtype=torch.float16)]
                # globals.sun_1_pred_relations[bytes.decode(mapping[i]['scenario_id'])] = int(argmax[i])
            if args.do_eval:
                return np.zeros([batch_size, self.future_frame_num, 2]), np.zeros(batch_size), None
            else:
                return loss.mean(), DE, None

        if 'variety_loss' in args.other_params:
            return self.variety_loss(mapping, hidden_states, batch_size, inputs, inputs_lengths, labels_is_valid, loss, DE, device, labels)
        elif 'goals_2D' in args.other_params:
            for i in range(batch_size):
                goals_2D = mapping[i]['goals_2D']

                self.goals_2D_per_example(i, goals_2D, mapping, lane_states_batch, inputs, inputs_lengths,
                                          hidden_states, labels, labels_is_valid, device, loss, DE)

            if 'tnt_square' in args.other_params:
                for i in range(0, batch_size, 2):
                    goals_2D, scores, gt_goal = mapping[i]['tnt_square.pack']
                    goals_2D_oppo, scores_oppo, gt_goal_oppo = mapping[i + 1]['tnt_square.pack']

                    gt_goal_idx = np.argmin(utils.get_dis_point_2_points(gt_goal, goals_2D))
                    gt_goal_idx_oppo = np.argmin(utils.get_dis_point_2_points(gt_goal_oppo, goals_2D_oppo))
                    gt_pair_idx = gt_goal_idx * len(goals_2D_oppo) + gt_goal_idx_oppo

                    normalizer = mapping[i]['normalizer']
                    normalizer_oppo = mapping[i + 1]['normalizer']

                    # Obtain goals in each other's local coordinate.
                    goals_2D_norm = utils_cython.get_normalized_points(
                        utils_cython.get_normalized_points(goals_2D, normalizer, reverse=True),
                        normalizer_oppo
                    )
                    goals_2D_oppo_norm = utils_cython.get_normalized_points(
                        utils_cython.get_normalized_points(goals_2D_oppo, normalizer_oppo, reverse=True),
                        normalizer
                    )

                    # Obtain goal states in each agent's own local frame, which will be used in evaluation.
                    goals_3D_raw = np.repeat(goals_2D[:, np.newaxis], goals_2D_oppo.shape[0], 1)
                    goals_3D_oppo_raw = np.repeat(goals_2D_oppo[np.newaxis], goals_2D.shape[0], 0)
                    # [num_joint_targets, 4]
                    goals_4D_raw = np.concatenate([goals_3D_raw, goals_3D_oppo_raw], -1).reshape((goals_2D.shape[0] * goals_2D_oppo.shape[0], -1))

                    if args.joint_target_type == "single":
                        # Use goal from a single agent's perspective.
                        goals_2D_oppo = goals_2D_oppo_norm
                    elif args.joint_target_type == "pair":
                        # Use goal from both agents' perspective.
                        goals_2D = np.concatenate([goals_2D, goals_2D_norm], -1)
                        goals_2D_oppo = np.concatenate([goals_2D_oppo, goals_2D_oppo_norm], -1)

                    # Combine goal states in designated coordinate.
                    goals_3D = np.repeat(goals_2D[:, np.newaxis], goals_2D_oppo.shape[0], 1)
                    goals_3D_oppo = np.repeat(goals_2D_oppo[np.newaxis], goals_2D.shape[0], 0)
                    # [num_joint_targets, 4]
                    goals_4D = np.concatenate([goals_3D, goals_3D_oppo], -1).reshape((goals_2D.shape[0] * goals_2D_oppo.shape[0], -1))

                    # Add joint likelihood.
                    scores_joint = (scores[:, np.newaxis] + scores_oppo[np.newaxis]).reshape((-1))

                    # Combine goals and compute embedding.
                    goals_5D = np.concatenate([goals_4D, scores_joint[:, np.newaxis]], -1)
                    goals_5D = self.tnt_square_goals_4D_encoder(torch.tensor(goals_5D, dtype=torch.float, device=device))

                    num_joint_goals = goals_5D.shape[0]

                    li = [
                        hidden_states[i, 0, :].unsqueeze(0).expand(num_joint_goals, -1),
                        hidden_states[i, 1, :].unsqueeze(0).expand(num_joint_goals, -1),
                        hidden_states[i + 1, 0, :].unsqueeze(0).expand(num_joint_goals, -1),
                        hidden_states[i + 1, 1, :].unsqueeze(0).expand(num_joint_goals, -1),
                        goals_5D
                    ]
                    joint_scores = self.tnt_square_decoder(torch.cat(li, dim=-1))
                    joint_scores = F.log_softmax(joint_scores.squeeze(-1), dim=-1)

                    if args.do_eval:
                        # Get goals (goals_4D_raw) in each agent's local frame.
                        utils.select_goal_pairs_by_NMS(mapping[i], mapping[i + 1], goals_4D_raw, utils.to_numpy(joint_scores),
                                                       args.nms_threshold,
                                                       mapping[i]['speed'],
                                                       mapping[i + 1]['speed'],
                                                       args)

                    loss[i] += F.nll_loss(joint_scores.unsqueeze(0), torch.tensor([gt_pair_idx], device=device))

            if args.do_eval:
                return self.goals_2D_eval(batch_size, mapping, labels, hidden_states, inputs, inputs_lengths, device)
            else:
                if args.visualize:
                    for i in range(batch_size):
                        predict = np.zeros((self.mode_num, self.future_frame_num, 2))

                        utils.visualize_goals_2D(mapping[i], mapping[i]['vis.goals_2D'], mapping[i]['vis.scores'],
                                                 self.future_frame_num,
                                                 labels=mapping[i]['vis.labels'],
                                                 labels_is_valid=mapping[i]['vis.labels_is_valid'],
                                                 predict=predict)

                return loss.mean(), DE, None
        else:
            assert False

    def get_scores(self, goals_2D_tensor: Tensor, inputs, hidden_states, inputs_lengths, i, mapping, device,
                   subgoal=None, subgoal_input=None):
        """
        :param goals_2D_tensor: candidate goals sampled from map (shape ['goal num', 2])
        :return: log scores of goals (shape ['goal num'])
        """
        if 'point_sub_graph' in args.other_params:
            goals_2D_hidden = self.goals_2D_point_sub_graph(goals_2D_tensor.unsqueeze(0), hidden_states[i, 0:1, :]).squeeze(0)
        else:
            goals_2D_hidden = self.goals_2D_mlps(goals_2D_tensor)

        goals_2D_hidden_attention = self.goals_2D_cross_attention(
            goals_2D_hidden.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0)

        shared_states = hidden_states[i, 0, :].unsqueeze(0).expand(goals_2D_hidden.shape)
        goals_decoder = self.goals_2D_decoder
        if subgoal is not None:
            shared_goal_states = subgoal_input.unsqueeze(0).repeat(goals_2D_hidden.shape[0], 1)
            shared_states = torch.cat([shared_states, shared_goal_states], -1)

            if subgoal == "3s":
                goals_decoder = self.goals_2D_decoder_3s
            elif subgoal == "5s":
                goals_decoder = self.goals_2D_decoder_5s
            else:
                raise NotImplementedError

        if 'raster' in args.other_params:
            raster_scale = 1
            raster_hidden = args.raster_image_hidden[i, ::raster_scale, ::raster_scale, :].reshape(-1, args.hidden_size)
            assert len(goals_2D_hidden) == len(raster_hidden), (len(goals_2D_hidden), len(raster_hidden))
            scores = goals_decoder(torch.cat([shared_states, goals_2D_hidden, goals_2D_hidden_attention, raster_hidden], dim=-1))
        else:
            if 'sub_to_decoder' in args.other_params:
                inf_hidden = args.inf_features[i]
                assert len(goals_2D_hidden) == len(inf_hidden), (len(goals_2D_hidden), len(inf_hidden))
                scores = goals_decoder(torch.cat([shared_states, goals_2D_hidden, goals_2D_hidden_attention, inf_hidden], dim=-1))
            else:
                scores = goals_decoder(torch.cat([shared_states, goals_2D_hidden, goals_2D_hidden_attention], dim=-1))

        scores = scores.squeeze(-1)
        scores = F.log_softmax(scores, dim=-1)
        return scores

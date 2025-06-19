#!/usr/bin/env python
import numpy as np
import os, argparse
import maze
import libpvrnn
import csv

np.set_printoptions(suppress=True)

act_encode = dict()
act_encode["center"] = [1, 0, 0, 0]
act_encode["down"] = [0, 1, 0, 0]
act_encode["left"] = [0, 0, 1, 0]
act_encode["right"] = [0, 0, 0, 1]

def softmax(x, temperature=1.0):
    e_x = np.exp(x/temperature)
    return e_x/e_x.sum()

### Commands for Experiment 1
### inference_discrete_aifg.py ../LibPvrnn/configs/2d_pftdsg.toml --input_goal_reached --save_subdir aifg

def main():
    ## Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="configuration file for PV-RNN")
    parser.add_argument("--max_steps", help="maximum steps to run for", type=int, default=3)
    parser.add_argument("--goal_pos", help="specify position of the (red) goal: left or right. Anything else is random", type=str, default="random")
    parser.add_argument("--input_goal_reached", help="append goal reached status to input", default=False, action="store_true")
    parser.add_argument("--save_subdir", help="save ER output in this subdirectory. Leave empty to not save", default="")
    parser.add_argument("--test_samples", "-n", help="number of times this runs", type=int, default=100)
    parser.add_argument("--er_samples", "-s", help="use multithreaded ER sampling", type=int, default=100)
    parser.add_argument("--sample_softmax_temp", help="enable softmax weighted sampling of policies, with distribution sharpness defined by temp. Set to 0 to use argmin", type=float, default=0.0)
    parser.add_argument("--randomize_seed", help="use a random seed for the RNG, otherwise use a fixed seed for reproducibility", default=False, action="store_true")
    parser.add_argument("--save_iter", help="calculate intermediate predictions as well", default=False, action="store_true")
    args = parser.parse_args()

    if not args.randomize_seed:
        rng = np.random.default_rng(seed=42)
    else:
        rng = np.random.default_rng()

    all_sample_loss = []
    all_sample_good = []
    all_sample_cs = []
    all_sample_goal = []
    all_good_action_fe = []
    all_bad_action_fe = []
    all_good_fe_avg = []
    all_good_fe_std =[]
    all_bad_fe_avg = []
    all_bad_fe_std = []
    all_selected_good = []
    all_selected = []

    goal_pos = None
    goal_pos_str = ""
    if args.goal_pos.lower() == "l" or args.goal_pos.lower() == "left":
        goal_pos = [1, 0]
        goal_pos_str = "left"
    elif args.goal_pos.lower() == "r" or args.goal_pos.lower() == "right":
        goal_pos = [0, 1]
        goal_pos_str = "right"
    else:
        if rng.random() < 0.5:
            goal_pos = [1, 0]
            goal_pos_str = "left"
        else:
            goal_pos = [0, 1]
            goal_pos_str = "right"

    if args.max_steps < 1:
        args.max_steps = 0
    else:
        args.max_steps -= 1

    ## Setup experiment
    world = maze.TMaze(goal_lr=goal_pos)

    for n in range(args.test_samples):
        print("**START OF SAMPLE", n)
        ## Setup PV-RNN
        sampling_mode = 0 # separate samples in this mode
        seed = n + (n*args.test_samples) # fixed rng seeds
        model_er = libpvrnn.PvrnnModel(verbose=False) # model for ER (future sampling)
        model_er.config.import_toml_config(args.config_file, task="online_error_regression") # load config
        if not args.randomize_seed:
            model_er.config.rng_seed = seed
        model_er.initialize_network(parallel_networks=args.er_samples)
        model_er.online_er_initialize(sampling_mode=sampling_mode)

        model_pg = libpvrnn.PvrnnModel(verbose=False) # model for plan generation (A optimization)
        model_pg.config.import_toml_config(args.config_file, task="planning") # load config
        if not args.randomize_seed:
            model_pg.config.rng_seed = seed
        model_pg.initialize_network(parallel_networks=args.er_samples)
        model_pg.online_plan_initialize(sampling_mode=sampling_mode)

        inputs = [[], [], []] # 3 steps
        inputs[0] = [act_encode["center"], world.clear]
        inputs[1] = [act_encode["down"], world.blue if goal_pos_str == "left" else world.green]
        inputs[2] = [act_encode[goal_pos_str], world.red]

        if args.input_goal_reached:
            inputs[0].append([0, 1])
            inputs[1].append([0, 1])
            inputs[2].append([1, 0])

        input_size = [s[1]-s[0] for s in model_er.config.dataset_slices] if model_er.config.dataset_slices else [model_er.config.output_size]
        mask_er = [list(np.ones(sz, dtype=int)) for sz in input_size]
        mask_pg = [list(np.ones((model_pg.config.window, sz), dtype=int)) for sz in input_size]

        # inputs is by timestep but ftarget is by modality
        ftarget = [[], []] if not args.input_goal_reached else [[], [], []]
        for t in range(len(inputs)):
            for m in range(len(ftarget)):
                ftarget[m].append(inputs[t][m])

        ## Main loop
        for i in range(0, args.max_steps):
            save_er = ""
            save_pg = ""
            if args.save_subdir != "":
                save_er = os.path.join(model_er.config.save_directory, args.save_subdir, "erstep" + str(i))
                save_pg = os.path.join(model_pg.config.save_directory, args.save_subdir, "pgstep" + str(i))

            net_input = [np.array(a, dtype=float) for a in inputs[i]] # fixed input

            ## Step 1: MS-ER
            _ = model_er.online_error_regression(input=net_input, mask=mask_er, save_path=save_er, save_iter_predictions=args.save_iter)
            # Download all samples
            samples_er = []
            for s in range(args.er_samples):
                samples_er.append(model_er.get_er_sequence(s))

            ## Step 2: Plangen        
            # When running the planner, the (future) target comes from ER, with mask = 1
            for tid, d in enumerate(model_pg.data):
                for dim, o in enumerate(d):
                    o.er_set_data(np.asarray(samples_er[tid][dim])[:model_pg.config.seq_len, :])
        
            model_pg.set_mask(mask_pg)
            model_pg.online_plan_generation_split_data(input=net_input, save_path=save_pg, plan_slide=(i >= model_er.config.window), save_iter=False)
            # Download all samples
            samples_plan = [] # (samples, olayers, len, dims)
            for s in range(args.er_samples):
                samples_plan.append(model_pg.get_er_sequence(s))

            ## Step 3: Evaluate plans
            # Build goal target and mask
            goal_target = ftarget
            goal_mask = [list(np.zeros((model_pg.config.window, sz), dtype=float)) for sz in input_size]
            # Unmask last sensory step as goal [modality, step, dim]
            goal_mask[1][-1] = np.ones_like(goal_mask[1][-1])
            if args.input_goal_reached: # Also unmask goal reached
                goal_mask[2][-1] = np.ones_like(goal_mask[2][-1])

            sample_loss = [0.0 for _ in range(args.er_samples)]
            for s in range(args.er_samples):
                goal_recloss = model_pg.er_compute_target_err(goal_target, goal_mask, s)
                goal_kld = model_pg.get_er_kld_sample(s)
                for o in range(len(model_pg.config.output_layers)):
                    # sample_loss[s] += model_pg.s_recerr[s][o][-1] # last iter
                    sample_loss[s] +=  np.mean(goal_recloss[o][i+1:])
                for l in range(model_pg.config.n_layers):
                    sample_loss[s] -= np.mean(goal_kld[l][i+1:]) # NB: inverted KLD
            all_sample_loss.append(sample_loss)
            all_sample_good.append([s[0][1][1] > 0.5 and s[1][2][2] > 0.5 for s in samples_plan])
            all_sample_cs.append([s[0][1][1] > 0.5 for s in samples_plan]) # visit down
            all_sample_goal.append([s[1][2][2] > 0.5 for s in samples_plan]) # saw goal
            # print("Sample losses =", sample_loss)
            if args.sample_softmax_temp == 0.0:
                selected_sample = np.argmin(sample_loss)
                print("Selected plan", selected_sample, "loss =", sample_loss[selected_sample])
            else:
                # Do EFE weighted sample selection
                sample_weights = softmax(-np.array(sample_loss), temperature=args.sample_softmax_temp)
                selected_sample = np.random.choice(args.er_samples, 1, p=sample_weights)[0]
                print("Selected plan", selected_sample, "prob =", sample_weights[selected_sample], "loss =", sample_loss[selected_sample])

            # Check the output
            all_selected.append(selected_sample)
            all_selected_good.append(samples_plan[selected_sample][0][1][1] > 0.5 and samples_plan[selected_sample][1][2][2] > 0.5) # our desired traj
            # print("Desired trajectory?", all_selected_good[-1])

            bad_action_fe = []
            good_action_fe = []
            for s in range(args.er_samples):
                desired_action = samples_plan[s][0][1][1] > 0.5 and samples_plan[s][1][2][2] > 0.5
                if desired_action:
                    good_action_fe.append(sample_loss[s])
                else:
                    bad_action_fe.append(sample_loss[s])

            all_good_action_fe.append(good_action_fe)
            all_bad_action_fe.append(bad_action_fe)

            good_fe_avg = None
            good_fe_std = None
            bad_fe_avg = None
            bad_fe_std = None

            if len(good_action_fe) > 0:
                good_fe_avg = np.average(good_action_fe)
                good_fe_std = np.std(good_action_fe)
            if len(bad_action_fe) > 0:
                bad_fe_avg = np.average(bad_action_fe)
                bad_fe_std = np.std(bad_action_fe)
            
            all_good_fe_avg.append(good_fe_avg)
            all_good_fe_std.append(good_fe_std)
            all_bad_fe_avg.append(bad_fe_avg)
            all_bad_fe_std.append(bad_fe_std)

            print("Good trajectories =", len(good_action_fe), "Good traj FE avg =", good_fe_avg, "stdev=", good_fe_std, "Bad traj FE avg =", bad_fe_avg, "stdev=", bad_fe_std)

            ## Step 4: Sync models with selected sample
            # model_pg.er_scatter_parameters(selected_sample) # sync plangen model
            # # sync ER model with plangen model
            # er_opt_A = np.asarray(model_pg.get_er_optimizer_A(selected_sample))[:,:model_pg.past_win_len+1,:] # trim future As
            # er_opt_A_1M = np.asarray(model_pg.get_er_optimizer_A_1M(selected_sample))[:,:model_pg.past_win_len+1,:]
            # er_opt_A_2M = np.asarray(model_pg.get_er_optimizer_A_2M(selected_sample))[:,:model_pg.past_win_len+1,:]
            # for s in range(args.er_samples):
            #     model_er.set_er_optimizer_parameters(s, er_opt_A, er_opt_A_1M, er_opt_A_2M)

            print("Step", i+1, "Loss rec =", [err[-1] for err in model_pg.recerr], "kld =", [err[-1] for err in model_pg.kld])
            # next_step = [samples_plan[selected_sample][o][model.past_win_len] for o in range(len(samples_plan[selected_sample]))]
            # print("Input", np.round(np.concatenate(net_input)), "Output", np.round(np.concatenate(next_step)))
        ## End of main loop
        print("**END OF SAMPLE", n)

    # Save CSVs
    for t in range(args.max_steps):
        with open("summary_t" + str(t) + ".csv", 'w', newline='') as summaryf:
            summaryw = csv.writer(summaryf, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            summaryw.writerow(["Good N", "CS N", "Goal N", "Good FE avg", "Good FE std", "Bad N", "Bad FE avg", "Bad FE std", "Selected good?", "Selected"])
            for n in range(args.test_samples):
                summaryw.writerow([len(all_good_action_fe[n*args.max_steps + t]), sum(all_sample_cs[n*args.max_steps + t]), sum(all_sample_goal[n*args.max_steps + t]), all_good_fe_avg[n*args.max_steps + t], all_good_fe_std[n*args.max_steps + t], len(all_bad_action_fe[n*args.max_steps + t]), all_bad_fe_avg[n*args.max_steps + t], all_bad_fe_std[n*args.max_steps + t], all_selected_good[n*args.max_steps + t], all_selected[n*args.max_steps + t]])

        with open("sample_loss_t" + str(t) + ".csv", 'w', newline='') as slossf, open("sample_good_t" + str(t) + ".csv", 'w', newline='') as sgoodf:
            slossw = csv.writer(slossf, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            sgoodw = csv.writer(sgoodf, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for n in range(args.test_samples):
                slossw.writerow(all_sample_loss[n*args.max_steps + t])
                sgoodw.writerow(all_sample_good[n*args.max_steps + t])

if __name__ == '__main__':
    main()

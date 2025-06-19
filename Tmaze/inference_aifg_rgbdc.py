#!/usr/bin/env python
import os, argparse, pickle
import numpy as np
import agent, maze, sim
import libpvrnn

np.set_printoptions(suppress=True)

def mad(data, axis=0):
    # Compute mean absolute deviation
    data = np.asarray(data)
    return np.median(np.absolute(data - np.mean(data, axis)), axis)

def softmax(x, temperature=1.0):
    e_x = np.exp(x/temperature)
    return e_x/e_x.sum()

def scaled_random(rng, min, max):
    return (max-min)*rng.random()+min

### Commands for Experiments 2 & 3
### inference_aifg_rgbdc.py ../LibPvrnn/configs/2d_pft_msgs_w2.toml --er_samples 100 --trials 100 --input_goal_sense --save_er_subdir aifg_exp2_fixedseed --save_all_subdir results/aifg_w2_exp2_fixedseed --epoch 200000 --max_step_size 0.5 --max_steps 25 --skip_plot --stop_early -1
### inference_aifg_rgbdc.py ../LibPvrnn/configs/2d_pft_rgbdc4s.toml --er_samples 100 --trials 100 --input_goal_sense --epoch 200000 --rgb_color --input_corner_sense --skip_plot

def main():
    ## Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="configuration file for PV-RNN")
    parser.add_argument("--epoch", help="epoch to load (-1 for last saved)", type=int, default=-1)
    parser.add_argument("--allow_teleport", help="ignore movement limits on the agent", default=False, action="store_true")
    parser.add_argument("--max_steps", help="maximum steps to run for", type=int, default=60)
    parser.add_argument("--max_step_size", help="maximum step size (delta) agent can take", type=float, default=0.2)
    parser.add_argument("--max_plan_switch_step", help="maximum step to allow plan switching. Late switching can be unstable", default=None)
    parser.add_argument("--goal_pos", help="specify position of the (red) goal: left or right. Anything else is random", type=str, default="random")
    parser.add_argument("--sticky_sensors", help="Allow the agent to remember the last seen color that isn't clear/white", default=False, action="store_true")
    parser.add_argument("--rgb_color", help="Use RGB color instead of softmax colors", default=False, action="store_true")
    parser.add_argument("--fgoal_modality", help="modality that is unmasked as the goal. -1 to disable", type=int, default=1)
    parser.add_argument("--fgoal_mask_offset", help="intially unmask target from this timestep (planner only)", type=int, default=0)
    parser.add_argument("--input_goal_reached", help="append goal reached status to input", default=False, action="store_true")
    parser.add_argument("--input_goal_map", help="append goal map (if available) to input", default=False, action="store_true")
    parser.add_argument("--input_goal_sense", help="append predicted goal to input", default=False, action="store_true")
    parser.add_argument("--input_corner_sense", help="append corner sensors to input", default=False, action="store_true")
    parser.add_argument("--delta_pos", help="use deltaXY movement. Also masks off movement error", default=False, action="store_true")
    parser.add_argument("--save_er_subdir", help="save ER output in this subdirectory. Leave empty to not save", default="")
    parser.add_argument("--save_all_subdir", help="save screenshots and state. Leave empty to not save", default="")
    parser.add_argument("--skip_plot", help="don't show the plot", default=False, action="store_true")
    parser.add_argument("--er_samples", help="use multithreaded ER sampling", type=int, default=1)
    parser.add_argument("--pg_samples", help="override thread count for plan generation, otherwise use the same number as ER", default=None)
    parser.add_argument("--trials", help="Number of trials to run, with success statistics at the end", type=int, default=1)
    parser.add_argument("--randomize_seed", help="use a random seed for the RNG, otherwise use a fixed seed for reproducibility", default=False, action="store_true")
    parser.add_argument("--save_iter", help="calculate intermediate predictions as well", default=False, action="store_true")
    parser.add_argument("--stop_early", help="end the trial if the robot has been at the goal for this long", type=int, default=3)
    parser.add_argument("--invert_cs", help="flip the color of the CS without changing the goal", default=False, action="store_true")
    args = parser.parse_args()

    if not args.randomize_seed:
        rng = np.random.default_rng(seed=41)
    else:
        rng = np.random.default_rng()

    successes = [0, 0]
    cs_checked = 0

    max_plan_switch_steps = int(args.max_plan_switch_step) if args.max_plan_switch_step is not None else args.max_steps # don't switch plans after this step

    # Obstacle parameters (0.5*0.5 square)
    # obstacle_coord_ranges = [([0.9, 1.0], [0.6, 0.8]), ([0.9, 1.0], [1.7, 1.9]), ([1.5, 1.6], [0.6, 0.8]), ([1.5, 1.6], [1.7, 1.9]), ([0.5, 1.0], [1.9, 2.0]), ([1.5, 2.0], [2.5, 2.6])]
    obstacle_coord_ranges = [([0.6, 0.9], [2.5, 2.5]), ([1.6, 1.9], [2.5, 2.5])] # upper wall

    for trial in range(args.trials):
        all_sample_loss = []
        cs_reached = False
        goal_pos = None
        if args.goal_pos.lower() == "l" or args.goal_pos.lower() == "left":
            goal_pos = [1, 0]
        elif args.goal_pos.lower() == "r" or args.goal_pos.lower() == "right":
            goal_pos = [0, 1]
        if goal_pos is None and not args.randomize_seed:
            goal_pos = [1, 0] if rng.random() < 0.5 else [0, 1]

        ## Setup experiment
        world = maze.TMaze(goal_lr=goal_pos, rgb_color=args.rgb_color)
        goal_pos = world.goal_lr
        if args.invert_cs:
            world.set_goal(goal_pos, invert_cs=True)
        goal_sense = world.red
        robot = agent.XYAgent(world=world, max_delta=args.max_step_size, max_sensor_range=0.75 if args.input_corner_sense else 0.0)
        pid = agent.SoftPID(Kp=0.85)
        goal = [robot.pos.copy(), goal_sense] # XY pos (not used for goal), color sense
        if args.input_corner_sense:
            goal += [[0.75, 0.75, 0.75, 0.75]] # this doesn't matter here
        if args.input_goal_reached:
            goal += [[1, 0]]
        elif args.input_goal_map:
            sensed_goal_pos = [0.5, 0.5] # initially not known!
            goal += [sensed_goal_pos]
        elif args.input_goal_sense:
            goal += [goal_sense] # here the prediction is just sensation

        ## Setup PV-RNN
        sampling_mode = 0 # separate samples in this mode
        seed = trial + (trial*args.trials) # fixed rng seeds
        model_er = libpvrnn.PvrnnModel(verbose=False) # model for ER (future sampling)
        model_er.config.import_toml_config(args.config_file, task="online_error_regression") # load config
        if not args.randomize_seed:
            model_er.config.rng_seed = seed
        model_er.initialize_network(parallel_networks=args.er_samples)
        model_er.online_er_initialize(sampling_mode=sampling_mode, epoch=args.epoch)

        model_pg = libpvrnn.PvrnnModel(verbose=False) # model for plan generation (A optimization)
        model_pg.config.import_toml_config(args.config_file, task="planning") # load config
        if not args.randomize_seed:
            model_pg.config.rng_seed = seed
        pg_samples = args.er_samples if args.pg_samples is None else int(args.pg_samples)
        model_pg.initialize_network(parallel_networks=pg_samples)
        model_pg.online_plan_initialize(sampling_mode=sampling_mode, epoch=args.epoch)

        input_size = [s[1]-s[0] for s in model_er.config.dataset_slices] if model_er.config.dataset_slices else [model_er.config.output_size]
        mask_er = [list(np.ones(sz)) for sz in input_size]
        if args.delta_pos:
            mask_er[0] = list(np.zeros_like(mask_er[0]))
        mask_pg = [list(np.ones((model_pg.config.window, sz))) for sz in input_size]

        ftarget = [list(np.zeros((model_pg.config.window, s))) for s in input_size]
        for t in range(len(ftarget)):
            for idx, g in enumerate(goal):
                ftarget[idx][t] = g

        ## Setup plot
        if not args.skip_plot:
            viz = sim.Sim2D(world, robot, paths=pg_samples, points=1, largetxt=True)

        if args.input_corner_sense:
            # Randomly place obstacle
            obstacle_range = obstacle_coord_ranges[rng.integers(0,len(obstacle_coord_ranges))]
            obstacle_pos = [scaled_random(rng, obstacle_range[0][0], obstacle_range[0][1]), scaled_random(rng, obstacle_range[1][0], obstacle_range[1][1])]
            print("Obstacle position:", obstacle_pos)
            world.set_obstacle(*obstacle_pos)
            if not args.skip_plot:
                viz.add_wall()

        # Save data
        xsave_subdir = None
        if len(args.save_all_subdir) > 0:
            xsave_subdir = os.path.join(args.save_all_subdir, "trial" + str(trial))
            os.makedirs(xsave_subdir, exist_ok=True)
        if not args.skip_plot:
            viz.draw(redraw_colors=True, redraw_walls=True, savepath=os.path.join(xsave_subdir, "0.png") if xsave_subdir is not None else "") # initial draw

        ttraj = []
        ptraj = []
        plans = []
        sensed = robot.sense
        ptraj.append(np.concatenate([robot.pos, sensed]))

        plan_past_win_size = model_pg.config.window
        selected_sample = 0
        selected_sample_loss = 99.9
        goal_reached_steps = 0
        ## Main loop
        for i in range(1, args.max_steps+1):
            save_er = ""
            save_pg = ""
            if args.save_er_subdir != "":
                save_er = os.path.join(model_er.config.save_directory, args.save_er_subdir, "trial" + str(trial), "erstep" + str(i))
                save_pg = os.path.join(model_pg.config.save_directory, args.save_er_subdir, "trial" + str(trial), "pgstep" + str(i))

            cur_pos = robot.pos.copy()
            goal_reached = np.allclose(robot.sense, goal_sense)
            if not args.sticky_sensors or (robot.sense != world.clear and sensed != robot.sense):
                sensed = robot.sense
            net_input = [np.array(cur_pos, dtype=np.float32), np.array(sensed, dtype=np.float32)]
            if args.input_corner_sense:
                net_input += [np.array(list(robot.corner_sensors.values()))]
            ttraj.append(np.concatenate([cur_pos, sensed]))
            if args.input_goal_reached:
                net_input += [np.array([1, 0] if goal_reached else [0, 1], dtype=np.float32)]
            elif args.input_goal_map:
                if sensed != world.clear:
                    sensed_goal_pos = world.goal_lr
                net_input += [sensed_goal_pos]
            elif args.input_goal_sense:
                net_input += [goal_sense]

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
            model_pg.online_plan_generation_split_data(input=net_input, save_path=save_pg, plan_slide=(i >= plan_past_win_size), save_iter=args.save_iter)

            # Download all samples
            samples_plan = [] # (samples, olayers, len, dims)
            for s in range(pg_samples):
                samples_plan.append(model_pg.get_er_sequence(s)) # download all samples
            vfactor = np.mean(mad([s[0] for s in samples_plan]))

            # Build goal target and mask (assume it's static for now)
            goal_target = ftarget
            goal_mask = [list(np.zeros((model_pg.config.window, sz), dtype=float)) for sz in input_size]

            if args.input_goal_reached: # Also unmask goal reached
                goal_mask[2][-1] = np.ones_like(goal_mask[2][-1])
            elif args.input_goal_sense:
                for t in range(i, model_pg.config.window):
                    goal_mask[2][t] = np.ones_like(goal_mask[2][t])
            else:
                # Unmask last sensory step as goal [modality, step, dim]
                goal_mask[1][-1] = np.ones_like(goal_mask[1][-1])

            ## Step 3: Evaluate plans
            sample_loss = [0.0 for _ in range(pg_samples)]
            sample_recerr = [0.0 for _ in range(pg_samples)]
            sample_kld = [0.0 for _ in range(pg_samples)]
            pg_opt_A = []
            pg_opt_A_1M = []
            pg_opt_A_2M = []
            for s in range(pg_samples):
                goal_recloss = model_pg.er_compute_target_err(goal_target, goal_mask, s)
                goal_kld = model_pg.get_er_kld_sample(s)
                for o in range(len(model_pg.config.output_layers)):
                    sample_recerr[s] += np.mean(goal_recloss[o][model_pg.past_win_len:])
                for l in range(model_pg.config.n_layers):
                    sample_kld[s] += np.mean(goal_kld[l][model_pg.past_win_len:])
                sample_loss[s] = (1.0/vfactor)*sample_recerr[s] - sample_kld[s]
                pg_opt_A.append([np.asarray(a) for a in model_pg.get_er_optimizer_A(s)])
                pg_opt_A_1M.append([np.asarray(a) for a in model_pg.get_er_optimizer_A_1M(s)])
                pg_opt_A_2M.append([np.asarray(a) for a in model_pg.get_er_optimizer_A_2M(s)])

            all_sample_loss.append((sample_recerr, sample_kld))
            print("Min loss", np.min(sample_loss), "Max loss", np.max(sample_loss), "Avg loss", np.average(sample_loss))

            # From sample loss, select plan
            sample_weights = np.ones_like(sample_loss) / len(sample_loss)
            candidate_sample = np.argmin(sample_loss)
            
            if i < max_plan_switch_steps and selected_sample_loss > sample_loss[candidate_sample]:
                selected_sample = candidate_sample
                selected_sample_loss = sample_loss[selected_sample]
                print(">Selected new plan", selected_sample, "with loss", selected_sample_loss)
            else:
                print(">Using current plan", selected_sample)

            ## Step 4: Sync models
            # (weighted) average
            wavg_opt_A = [sample_weights[0]*p for p in pg_opt_A[0]]
            wavg_opt_A_1M = [sample_weights[0]*p for p in pg_opt_A_1M[0]]
            wavg_opt_A_2M = [sample_weights[0]*p for p in pg_opt_A_2M[0]]
            for s in range(1, pg_samples):                
                wavg_opt_A = [a+sample_weights[s]*pg_opt_A[s][idx] for idx,a in enumerate(wavg_opt_A)]
                wavg_opt_A_1M = [a+sample_weights[s]*pg_opt_A_1M[s][idx] for idx,a in enumerate(wavg_opt_A_1M)]
                wavg_opt_A_2M = [a+sample_weights[s]*pg_opt_A_2M[s][idx] for idx,a in enumerate(wavg_opt_A_2M)]

            for s in range(args.er_samples):
                model_pg.set_er_optimizer_parameters(s, wavg_opt_A, wavg_opt_A_1M, wavg_opt_A_2M)

            plans.append(samples_plan[selected_sample])
            next_step = [plans[-1][o][model_pg.past_win_len] for o in range(len(plans[-1]))]
            ptraj.append(np.concatenate(next_step))

            # MOVE ROBOT
            robot.move(pid.move(cur_pos, next_step[0]), force=args.allow_teleport, delta_pos=args.delta_pos)

            print("Step", i, "ER loss rec =", [err[-1] for err in model_er.recerr], "kld =", [err[-1] for err in model_er.kld])
            print("Step", i, "Plan loss rec =", [err[-1] for err in model_pg.recerr], "kld =", [err[-1] for err in model_pg.kld])

            if world.convert_color(sensed) == "blue" or world.convert_color(sensed) == "green":
                cs_reached = True

            # Plot this step
            plot_samples = []
            plot_samples.append([[x[0] for x in ttraj], [x[1] for x in ttraj]])
            for s in range(pg_samples):
                plot_samples.append([np.asarray(samples_plan[s][0])[:,0], np.asarray(samples_plan[s][0])[:,1]])

            if not args.skip_plot:
                viz.draw_agent.set_facecolor(world.convert_color(sensed) if not goal_reached else "yellow")
                viz.draw(paths=plot_samples, highlight_path=selected_sample+1, dashed_path=0, points=[next_step[0]], savepath=os.path.join(xsave_subdir, str(i) + ".png") if xsave_subdir is not None else "")

            if len(args.save_all_subdir) > 0:
                with open(os.path.join(xsave_subdir, "samples_er_step" + str(i) + ".pkl"), "wb") as ofile:
                    pickle.dump(samples_er, ofile)
                with open(os.path.join(xsave_subdir, "samples_plan_step" + str(i) + ".pkl"), "wb") as ofile:
                    pickle.dump(samples_plan, ofile)
                with open(os.path.join(xsave_subdir, "all_sample_loss_step" + str(i) + ".pkl"), "wb") as ofile:
                    pickle.dump(all_sample_loss, ofile)
                with open(os.path.join(xsave_subdir, "all_sample_agent" + str(i) + ".pkl"), "wb") as ofile:
                    all_sample_agent = {"goal_pos":goal_pos, "vfactor":vfactor, "selected_sample":selected_sample, "robot_pos":robot.pos, "robot_sense":robot.sense, "past_win_len":model_pg.past_win_len}
                    if args.input_corner_sense:
                        all_sample_agent["corner_sensors"] = list(robot.corner_sensors.values())
                    pickle.dump(all_sample_agent, ofile)
                with open(os.path.join(xsave_subdir, "all_sample_ttraj" + str(i) + ".pkl"), "wb") as ofile:
                    pickle.dump(ttraj, ofile)
            if goal_reached:
                print("**Goal reached")
                goal_reached_steps += 1
                if args.stop_early > 0 and goal_reached_steps >= args.stop_early:
                    print("**Ending trial early")
                    break
            else:
                goal_reached_steps = 0

        if cs_reached:
            cs_checked += 1
        if goal_reached:
            if goal_pos == [1, 0]:
                successes[0] += 1
            elif goal_pos == [0, 1]:
                successes[1] += 1
            else:
                print("Unknown goal sig!")
        if not args.skip_plot:
            viz.close()
        

    # End of trials
    print("## Success rate: " + str((successes[0]+successes[1])/args.trials) + " split: " + str(successes[0]) + " | " + str(successes[1]) + " CS rate: " + str(cs_checked/args.trials) + " ##")

if __name__ == '__main__':
    main()

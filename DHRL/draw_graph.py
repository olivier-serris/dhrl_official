import os
import json
import argparse
import gymnasium as gym
from main import launch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from DHRL.rl.algo.dhrl import Algo
import numpy as np


def plot_interaction_step(ax: Axes, obs, intermediate_goal, intermediate_edges, title):

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    observation = obs["observation"]
    desired_goal = obs["desired_goal"]

    ax.scatter(observation[0], observation[1], color="green")
    ax.scatter(intermediate_goal[0], intermediate_goal[1], color="blue")
    ax.scatter(desired_goal[0], desired_goal[1], color="red")

    lc = LineCollection(intermediate_edges, alpha=1, zorder=0)
    ax.add_collection(lc)
    ax.set_title(title)


def see_agent(
    env: gym.Env, agent: Algo, eval_id, save_folder, save_frames=False, render=False
):

    env = gym.make(env.unwrapped.spec.id, render_mode="human" if render else None)
    eval_options = env.get_wrapper_attr("eval_options")
    options = {k: eval_options[k][eval_id] for k in eval_options}
    obs, info = env.reset(options=options)

    if save_frames:
        ax = plt.subplot()

    agent.agent_api.reset(None, obs)
    for t in range(300):
        if t == 150:
            pass

        obs = {k: np.copy(v) for k, v in obs.items()}
        _, action = agent.agent_api(None, obs)
        if save_frames:
            nodes = agent.graphplanner.landmarks[
                agent.graphplanner.waypoint_vec[agent.graphplanner.waypoint_idx :]
            ]
            nodes = np.concatenate(
                [
                    obs["achieved_goal"].reshape(1, 2),
                    nodes,
                    agent.waypoint_subgoal.reshape(1, 2),
                ]
            )
            edges = np.concatenate(
                [nodes[:-1,].reshape(1, -1, 2), nodes[1:].reshape(1, -1, 2)]
            )
            title = f"{t}"
            ax.clear()
            plot_interaction_step(
                ax,
                obs,
                agent.waypoint_subgoal,
                edges,
                title=title,
            )
            plt.savefig(os.path.join(save_folder, title))

        # action = agent.low_agent.get_actions(obs["observation"], obs["desired_goal"])
        obs, reward, terminated, truncated, info = env.step(action)
        # done = np.logical_or(terminated, truncated)
        done = terminated
        if done:
            obs, info = env.reset(options=options)
            agent.agent_api.reset(None, obs)
    env.close()
    if save_frames:
        plt.close()


def plot_graph_path(ax: Axes, nodes, path, edges):
    ax.scatter(nodes[:, 0], nodes[:, 1], color="black")

    lc = LineCollection(edges, alpha=0.1, zorder=0)
    ax.add_collection(lc)
    ax.plot(path[:, 0], path[:, 1], color="red")
    ax.scatter(path[-1, 0], path[-1, 1], color="red")


def load_agent_and_env(path):
    hp_path = os.path.join(path, "config.json")
    with open(hp_path, "r") as f:
        args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)
    agent = launch(args)
    agent.load_all(os.path.join(path, "state"))
    return agent, agent.test_env


def save_graph(agent: Algo, draw_env, observation, save_path):

    agent.graphplanner.find_path(
        observation["observation"], observation["desired_goal"]
    )
    waypoint_ids = agent.graphplanner.waypoint_vec
    landmarks = agent.graphplanner.landmarks
    waypoint = np.concatenate(
        [
            observation["achieved_goal"].reshape(1, -1),
            landmarks[waypoint_ids],
            observation["desired_goal"].reshape(1, -1),
        ],
        # axis=-1,
    )

    edges_ids = np.array(agent.graphplanner.graph.edges)
    edges = landmarks[edges_ids]
    ax = plt.subplot()
    if draw_env:
        draw_env(ax)
    plot_graph_path(
        ax,
        landmarks,
        waypoint,
        edges,
    )
    plt.savefig(save_path)
    plt.close()


def main():
    path = "exp/PointMaze_UMaze-v3/Dec-2-11-13-53"
    agent, env = load_agent_and_env(path)
    eval_options = env.get_wrapper_attr("eval_options")
    n_eval = eval_options[next(iter(eval_options))].shape[0]
    try:
        draw_env = env.get_wrapper_attr("plot")
    except:
        draw_env = None
    folder = os.path.join(path, "plots")

    # options = {k: v[n_eval - 1] for k, v in eval_options.items()}
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(n_eval):
        options = {k: v[i] for k, v in eval_options.items()}
        for j in range(5):
            agent.way_to_subgoal = 0
            agent.graphplanner.graph_construct(-1)
            observation, infos = env.reset(options=options)
            save_graph(
                agent, draw_env, observation, os.path.join(folder, f"eval_{i}_{j}")
            )
            interaction_folder = os.path.join(folder, f"traj_{i}_{j}")
            if not os.path.exists(interaction_folder):
                os.makedirs(interaction_folder)
            see_agent(
                env,
                agent,
                i,
                save_folder=interaction_folder,
                save_frames=True,
            )


if __name__ == "__main__":
    main()

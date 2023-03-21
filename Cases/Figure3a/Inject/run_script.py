import numpy as np
import sys
from tqdm import tqdm

sys.path.append("../../../")

from Cases.Figure3a.Inject.set_environment import Environment


def main():
    # Set simulation final time
    final_time = 3.0

    env = Environment(
        final_time,
        COLLECT_DATA_FOR_POSTPROCESSING=True,
        k_straight_straight_connection_spring_scale=0.166,
        k_straight_straight_connection_contact_scale=2.0,
        k_ring_ring_spring_connection_scale=0.241,
        k_ring_straight_spring_connection_scale=1.0,
        k_ring_straight_spring_torque_connection_scale=5,
        k_ring_straight_contact_connection_scale=10.0,
        k_ring_helical_spring_connection_scale=1.0,
        k_ring_helical_contact_connection_scale=96,
    )

    # Do multiple simulations for learning, or control
    for i_episodes in range(1):
        # Reset the environment before the new episode and get total number of simulation steps
        total_steps, systems = env.reset()

        # Simulation loop starts
        time = np.float64(0.0)
        user_defined_condition = False
        reward = 0.0

        LOAD_FROM_RESTART = False
        SAVE_DATA_RESTART = True
        restart_file_location = "tapered/data/"
        if LOAD_FROM_RESTART:
            # env.step([], time)  # Before load run simulation 1 step
            time = env.load_state(restart_file_location, verbose=True)

        for i_sim in tqdm(range(total_steps)):
            activation = (
                []
            )  # segment_activation_function(number_of_muscle_segments, time)

            # Do 200 simulation step. Number of simulation steps can be changed in Environment class.
            time, systems, done = env.step(activation, time)

            if user_defined_condition == True:
                print(" User defined condition satisfied, exit simulation")
                print(" Episode finished after {} ".format(time))
                break

            # If done=True, NaN detected in simulation.
            # Exit the simulation loop before, reaching final time
            if done:
                print(" Episode finished after {} ".format(time))
                break

        print("Final time of simulation is : ", time)
        # Simulation loop ends

        # Post-processing
        # Make a video of octopus for current simulation episode. Note that
        # in order to make a video, COLLECT_DATA_FOR_POSTPROCESSING=True
        env.post_processing(
            filename_video="tapered_arm_with_helical_rods.mp4",
            # The following parameters are optional
            x_limits=(-200.0, 200.0),  # Set bounds on x-axis
            y_limits=(-100, 300),  # Set bounds on y-axis
            z_limits=(-200.0, 200.0),  # Set bounds on z-axis
            dpi=100,  # Set the quality of the image
            vis3D=True,  # Turn on 3D visualization
            vis2D=True,  # Turn on projected (2D) visualization
        )

        # Save simulation states for restart
        if SAVE_DATA_RESTART:
            env.save_state(restart_file_location, time, verbose=True)

    return env


if __name__ == "__main__":
    env = main()

__doc__ = """This script is computing and plotting topological quantities. 
    In order to run this file, please first run simulations."""
import numpy as np
import sys

sys.path.append("../")
sys.path.append("../../")
from KnotTheory.twist import compute_twist
from KnotTheory.link import compute_link
from KnotTheory.writhe import compute_writhe
from KnotTheory.pre_processing import compute_additional_segment


my_data = np.load("octopus_arm_test.npz")

n_rods, time_shape, _, n_elems = my_data["straight_rods_position_history"].shape
time = my_data["time"]

position_history = my_data["straight_rods_position_history"][0, :, :, :]
radius_history = my_data["straight_rods_radius_history"][0, :, :]
director_history = my_data["straight_rods_director_history"][0, :, :, :, :]
base_length = my_data["straight_rods_length_history"][0, 0, :].sum()

normal_history = -director_history[:, 1, :, :]

segment_length = 10 * base_length

type_of_additional_segment = "next_tangent"
# type_of_additional_segment = "net_tangent"
# type_of_additional_segment = "end_to_end"
# type_of_additional_segment = "average_tangent"

total_twist, local_twist = compute_twist(position_history, normal_history)

total_link = compute_link(
    position_history,
    normal_history,
    radius_history,
    segment_length,
    type_of_additional_segment,
)

total_writhe, segment_writhe = compute_writhe(
    position_history, segment_length, type_of_additional_segment
)

total_length_history = np.sum(my_data["straight_rods_length_history"][0, :, :], axis=1)

center_line, beginning_direction, end_direction = compute_additional_segment(
    position_history, segment_length, type_of_additional_segment
)

import matplotlib

matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt

# # Plotting
plt.rcParams.update({"font.size": 22})
fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

axs = []
axs.append(plt.subplot2grid((1, 1), (0, 0)))
axs[0].plot(
    time[1:],
    total_twist[1:],
    label="twist",
)
axs[0].plot(
    time[1:],
    total_writhe[1:],
    label="writhe",
)
axs[0].plot(
    time[1:],
    total_link[1:],
    label="link",
)
axs2 = axs[0].twinx()
error = np.abs(total_link - (total_writhe + total_twist))
axs2.semilogy(
    time[1:],
    error[1:],
    "--",
    c="r",
    label="Lk - (Wr+Tw)",
)
axs[0].set_xlabel("time [s]", fontsize=20)
axs[0].set_ylabel("link-twist-writhe", fontsize=20)
axs2.set_ylabel("Lk - (Wr+Tw)", fontsize=20)
plt.tight_layout()
fig.align_ylabels()
fig.legend(prop={"size": 20})
fig.savefig("link_twist_writhe.png")
plt.close(plt.gcf())

import numpy as np
import argparse
import json 

import sys

from Utils.graph_operations import get_graph_from_file
parser = argparse.ArgumentParser()

parser.add_argument("barabasi", help = "Barabasi scoring metric")
parser.add_argument("-n", "--no_pred_edges", type = int,  help = "Number of Edges to be Predicted")
parser.add_argument("-e", "--edges_to_ignore", help = "Edges to Ignore")
parser.add_argument("-x", "--x_state_file" , help = "State Embedding File")
parser.add_argument("-d", "--dict_mat_to_label", help = "Matrix to String Label")
parser.add_argument("-o", "--output", help = "Output Predicted Edge Lists")
parser.add_argument("-b", "--beta1", type = float, default = 1.0, help = "Beta1 parameter")
parser.add_argument("-g", "--gamma1", type = float, default = 1.0, help = "Gamma1 parameter")
parser.add_argument("-t", "--beta2", type = float, default = 0.0, help = "Beta2 parameter")
parser.add_argument("-r", "--gamma2", type = float, default = 10.0, help = "Gamma2 parameter")
parser.add_argument("-c", "--cons", type = float, default = 1.0, help = "Constant parameter")

args = parser.parse_args()

barabasi_file = args.barabasi
no_pred       = args.no_pred_edges
x_state       = args.x_state_file
e_to_ignore   = args.edges_to_ignore
f_m_to_l      = args.dict_mat_to_label
output_f      = args.output
beta1         = args.beta1
gamma1        = args.gamma1
beta2         = args.beta2
gamma2        = args.gamma2
cons          = args.cons

edge_list = get_graph_from_file(e_to_ignore)
m_l_dict  = None
with open(f_m_to_l, "r") as m_l_f:
    m_l_dict = json.load(m_l_f)

#####################
## Edges to Ignore ##
#####################

e_dict_to_ignore = {}
with open(e_to_ignore, "r") as ef:
    for line in ef:
        words = line.lstrip().rstrip().split()[ : 3]
        e_dict_to_ignore[(words[0], words[1])] = float(words[2])


#####################
## Barabasi Metric ##
#####################

b_metric_dict = {}
with open(barabasi_file, "r") as bf:
    for line in bf:
        p, q, wt = line.lstrip().rstrip().split()
        wt = float(wt)
        b_metric_dict[(p, q)] = wt

########################
## Getting State file ##
########################

state_file = np.load(x_state)
no_nodes   = state_file.shape[0]

#########################
## Compute l2 distance ##
#########################

l2_dist = np.matmul(state_file, state_file.T)
diag_m  = np.matmul(np.diagonal(l2_dist).reshape(no_nodes, 1), np.ones((1, no_nodes)))
l2_dist = diag_m + diag_m.T - 2 * l2_dist
 
edges_with_lists = []

#####################
## Compute Ranking ##
#####################

for i in range(no_nodes):
    for j in range(i):
        i_label = m_l_dict[str(i)]
        j_label = m_l_dict[str(j)]

        if (i_label, j_label) in e_dict_to_ignore or (j_label, i_label) in e_dict_to_ignore:
            continue
        b_metric = 0
        
        b_metric_key = (i_label, j_label) if (i_label, j_label) in b_metric_dict else (j_label, i_label) 
        if (b_metric_key in b_metric_dict):
            b_metric = b_metric_dict[b_metric_key]
        exp_dse_metric = np.exp(beta1 / float(1 + gamma1 * l2_dist[i, j]))
        exp_b_metric = np.exp(-beta2 / float(1 + gamma2 * b_metric))
        updated_metric_i_j = exp_dse_metric * b_metric + cons * exp_b_metric  / l2_dist[i, j]

        edges_with_lists.append((i_label, j_label, updated_metric_i_j))

edges_with_lists_sorted = sorted(edges_with_lists, key = lambda x : x[2], reverse = True)

if no_pred > 0:
    edges_with_lists_sorted = edges_with_lists_sorted[ : no_pred]

with open(output_f, "w") as of:
    str_to_wrt = ""
    for edge in edges_with_lists_sorted:
        str_to_wrt += "{} {} {}\n".format(edge[0], edge[1], edge[2])
    str_to_wrt = str_to_wrt.lstrip().rstrip()
    of.write(str_to_wrt)




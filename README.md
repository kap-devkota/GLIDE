
# GLIDE - Global and Local Integrated Diffused Embedding

GLIDE is a novel embedding method that embeds an undirected graph (which is connected) by utilizing the global spectral properties of the graph, and combines it with local graph properties to produce a score between prospective links. The link scores denote the confidence of the link being a missing edge in the graph.

## File Organization

### gen_x_state.py
GLIDE requires the user to perform the graph embedding before generating the ranking results. The file *gen_x_state.py* is used to perform the embedding. It can be invoked as:

> $ python gen_x_state.py [-h] [-l lap] [-d deg] [-n n_proj] [-j json] [-p gamma] [-i] [-z] ppi_network out_file

The parameters description is illustrated below:
1. **ppi_network (Required)**: The graph file, which is undirected and connected. It should be a text file of the form:
    >node1  node2 weight12
    >node3  node4 weight34 
2. **out_file (Required)**: The output numpy binary matrix of size *Nxk*, *N* being the number of nodes in the graph, and *k* being the projection size.
3. **[-l lap]** (Optional): If set, outputs the graph laplacian in a file named  **lap**.
4. **[-d deg]** (Optional): If set, outputs the diagonal degree matrix of the graph in a file named **deg**.
5. **[-n n_proj]** (Optional): Default value: -1. The size of projection, given by *k*. When *k < 0*, it means the number of embedding *k* is equal to *N*.
6. **[-j json] (Required)**: It is the output json file that maps the row index of the embedding matrix (given by **ppi_network**) into the node label. Here, the matrix row index in the embedding, where each row represents a particular node, is the key and the actual name of the node is the value of the json dictionary.
7. **[-p gamma]** (Optional): The gamma value that is a DSE-gamma embedding parameter. Default value = 1
8. **[-i]** (Optional): If set, the embedding is Coifman embedding. Else, it is a DSE-gamma embedding
9. **[-z]** (Optional): If set, the DSE-gamma is normalized by the steady state vector. 

Example:
> python gen_x_state.py -p 0.1 -j *Embedding/d_3_nodelist.json* *Datasets/dream_3.txt* *Embedding/embed_d_3_gamma_0.1.npy*

Here, the graph file is *Embedding/d_3_nodelist.json*. The output files are *Embedding/d_3_nodelist.json*, which is a index to label json dictionary file, and "Embedding/embed_d_3_gamma_0.1.npy", which is a numpy matrix file.

### combine_common_dse.py
This python file, which  is used to perform GLIDE(CW) ranking, is invoked as: 
> python combine_common_dse.py [-n no_pred_edges] [-e e_to_ignore] [-x dse_embed] [-d json_dict] [-o output_rank] [-b beta1] [-g gamma1] [-t beta2] [-r gamma2] [-c cons]

The parameters description is illustrated below:
1. **[-n no_pred_edges]** (Optional): How many top scoring links to output. Default = -1, which means to output all the links
2. **[-e e_to_ignore]** : This denotes the file containing the already-present links in the graph which is to be ignored.
3. **[-x dse_embed] (Required)**: This file represents the DSE-gamma embedding file.
4. **[-d json_dict] (Required)**: This file represents the input index to label dictionary json file.
5. **[-o output_rank] (Required)**: This file represents the output rank file. The file outputs the top scoring links in descending order, in text format, where each line in the file represents a link in the form: **node1 node2 score**.
6. **[-b beta1]**: beta1 score: Default is 1.
7. **[-g gamma1]**: gamma1 score: Default is 1.
8. **[-t beta2]** (Optional): beta2 score: Default is 0. Not to be changed for general purpose.
9. **[-r gamma2]** (Optional): gamma2 score: Default is 10. Not to be changed for general purpose.
10. **[-c cons]**: cons score: Default is 1.

Example:
> python combine_common_dse.py  -n 1000  -e *Datasets/dream_3.txt* -x *Embedding/embed_d_3_gamma_0.1.npy* -d *Embedding/d_3_nodelist.json* -o Rank/d3_ranks.txt -b 0.1 -g 1000 -t 0.0 -r 1.0 -c 0.0001

### combine_loc_dse.py
This python file, which  is used to perform GLIDE(L3) ranking, is invoked as: 
> python combine_loc_dse.py [-n no_pred_edges] [-e e_to_ignore] [-x dse_embed] [-d json_dict] [-o output_rank] [-b beta1] [-g gamma1] [-t beta2] [-r gamma2] [-c cons] loc_rank_file

The parameters description is illustrated below:
1. **[-n no_pred_edges]** (Optional): How many top scoring links to output. Default = -1, which means to output all the links
2. **[-e e_to_ignore]** : This denotes the file containing the already-present links in the graph which is to be ignored.
3. **[-x dse_embed] (Required)**: This file represents the DSE-gamma embedding file.
4. **[-d json_dict] (Required)**: This file represents the input index to label dictionary json file.
5. **[-o output_rank] (Required)**: This file represents the output rank file. The file outputs the top scoring links in descending order, in text format, where each line in the file represents a link in the form: **node1 node2 score**.
6. **[-b beta1]**: beta1 score: Default is 1.
7. **[-g gamma1]**: gamma1 score: Default is 1.
8. **[-t beta2]**: beta2 score: Default is 0. Not to be changed for general purpose.
9. **[-r gamma2]**: gamma2 score: Default is 10. Not to be changed for general purpose.
10. **[-c cons]**: cons score: Default is 1.
11. **loc_rank_file** **(Required)**: The local rank file, (Can be L3 scoring rank file), to be inputted.

Example:
> python combine_loc_dse.py  -n 1000  -e *Datasets/dream_3.txt* -x *Embedding/embed_d_3_gamma_0.1.npy* -d *Embedding/d_3_nodelist.json* -o Rank/d3_ranks.txt -b 0.1 -g 1000 -t 0.0 -r 1.0 -c 0.0001 *Local_Rank/d_3_l3.txt*
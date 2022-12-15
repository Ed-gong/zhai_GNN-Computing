import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
#from dgl import DGLGraph
from dgl.data import register_data_args, load_data
#import dgl.function as fn
#from dgl.function import TargetCode
#import dgl.backend as backend
#import dgl.utils as utils 
#import dgl
from torch.utils.cpp_extension import load
gnc = load(
    name="gnncompile",
    sources=[
        "kernel.cpp",
        "kernel_generated.cu",
        "../src/util.cu",
        "../src/data.cu"
    ]
    #extra_compile_args=['-g']
    )
    
def gcn_layer_dgl_no_graph(feat, weight, in_feat_len, out_feat_len):
    feat2 = torch.mm(feat, weight)
    output = F.relu(feat2)
    torch.cuda.synchronize()
    return output
def gat_layer_dgl_no_graph(feat, weight, attn_l, attn_r, in_feat_len, out_feat_len):
    feat2 = torch.mm(feat, weight)
    att_l = torch.mm(feat2, attn_l)
    att_r = torch.mm(feat2, attn_r)
    torch.cuda.synchronize()
    return feat2 

def gcn_layer_dgl(feat, weight, in_feat_len, out_feat_len):
    feat2 = torch.mm(feat, weight)
    g.ndata['ft'] = feat2
    g.update_all(fn.copy_src(src='ft', out='m'),
                   fn.sum(msg='m', out='ft'))
    output = F.relu(g.dstdata['ft'])
    torch.cuda.synchronize()
    return output
def gat_layer_dgl(feat, weight, attn_l, attn_r, in_feat_len, out_feat_len):
    feat2 = torch.mm(feat, weight)
    att_l = torch.mm(feat2, attn_l)
    att_r = torch.mm(feat2, attn_r)
    g.srcdata.update({'ft': feat2, 'el': att_l})
    g.dstdata.update({'er': att_r})
    g.apply_edges(fn.u_add_v('el', 'er', 'e'))
    e = torch.exp(F.leaky_relu(g.edata.pop('e'), 0.1))

    cont = utils.to_dgl_context(e.device)
    gidx = g._graph.get_immutable_gidx(cont)
    e_sum = backend.copy_reduce("sum", gidx, TargetCode.EDGE, e, num_v)
    att = backend.binary_reduce('none', 'div', gidx, TargetCode.EDGE, TargetCode.DST, e, e_sum, n_edges)
    g.edata['a'] = att
    g.update_all(fn.u_mul_e('ft', 'a', 'm'),
                    fn.sum('m', 'ft'))
    output = g.dstdata['ft']
    torch.cuda.synchronize()
    return output
    
# def gat_layer_our(feat, output_feat, weight, attn_lr, att_mid, val_mid, in_feat_len, out_feat_len, at):
def gat_layer_our(at_gat, at, feat, output_feat, weight, weight_lr, num_v, num_e, num_heads):

    feat2 = torch.mm(feat, weight)
    # they only support 1 head, 3 head needed the 3d tensor
    #feat2 = torch.rand(num_v, num_heads, output_dim).cuda()
    print("feat2 size", feat2.size())
    att_lr = torch.mm(feat2, weight_lr)
    # 2 means both left and right
    #feat2 = torch.rand(num_v, num_heads, output_dim).cuda()
    #att_lr = torch.rand(num_v, num_heads, 2).cuda()
    print("att_lr size", att_lr.size())
    val_mid = torch.zeros(num_e, num_heads).cuda()
    att_mid = torch.zeros(num_v, num_heads, 1).cuda()
    print("begin to run the kernel")
    time0 = time.time()
    gnc.gat_run_u_add_v(at_gat, att_lr, val_mid, 128)
    torch.cuda.synchronize()
    print("our sddmme kernel time {} ".format( time.time() - time0 ))
    print("sddmme kernel size attention_lr(input), and output", att_lr.size(), val_mid.size())

    val_mid = torch.exp(F.leaky_relu(val_mid, 0.1))
    torch.cuda.synchronize()

    time0 = time.time()
    gnc.gat_run_add_to_center(at_gat, val_mid, att_mid, 128)
    torch.cuda.synchronize()
    print("our spmmw2d kernel time {} ".format( time.time() - time0 ))
    print("spmmw2d kernel size", val_mid.size(), att_mid.size())

    time0 = time.time()
    gnc.gat_run_div_each(at_gat, att_mid, val_mid, 128)
    torch.cuda.synchronize()
    print("our sddmm kernel time {} ".format( time.time() - time0 ))
    print("sddmm kernel size", att_mid.size(), val_mid.size())

    time0 = time.time()
    gnc.gcn_update_val(at, val_mid)
    torch.cuda.synchronize()
    gnc.gcn_run(at, feat2, output_feat, 128, 1)
    torch.cuda.synchronize()
    print("our spmmw_op kernel time {}".format(time.time() - time0 ))
    print("size", val_mid.size(), feat2.size(), output_feat.size())
    print("spmmw_op:feat_output, updated edge", output_feat, val_mid)
    return output_feat


def test_spmmw_op(at, output_feat, num_v, num_e, num_heads, output_dim):
    #feat2 = torch.mm(feat, weight)
    feat2 = torch.ones(num_v, output_dim).cuda();
    val_mid = torch.ones(num_e, num_heads).cuda()
    
    #feat1 = torch.load("/home/pkumar/src/GraphPy_workflow/new_script/data_save/spmmw_op2d_X.pt").cuda()
    #feat1 = torch.load('/home/ygong07/exp/GraphPy_workflow/new_script/reddit_spmmw_op2d_X.pt').cuda()
    
    #feat2 = feat1.view(num_v, output_dim) 
    #val_mid = torch.load("/home/pkumar/src/GraphPy_workflow/new_script/data_save/spmmw_op2d_Y.pt").cuda()
    
    print(feat2.size(), val_mid.size());
    #print("spmmw_op before:feat_output", output_feat)
    time0 = time.time()
    gnc.gcn_update_val(at, val_mid)
    torch.cuda.synchronize()
    gnc.gcn_run(at, feat2, output_feat, 128, 1)
    torch.cuda.synchronize()
    print("our spmmw_op kernel time {}".format(time.time() - time0 ))
    print("size", val_mid.size(), feat2.size(), output_feat.size())
    print("spmmw_op after:feat_output", output_feat)
    result = torch.load('/home/ygong07/exp/GraphPy_workflow/new_script/reddit_spmmw_op2d_result.pt').cuda()
    print("spmmw2d results are the same", torch.all(output_feat.eq(result)))
    return output_feat


def gat_layer_our2(feat, output_feat, att_lr, att_mid, val_mid, in_feat_len, out_feat_len, at):
    gnc.gat_run(at_gat, feat, att_lr, output_feat, 128, 1)
    # gnc.gat_run_u_add_v(at_gat, att_lr, val_mid, 128)
    # val_mid = torch.exp(F.leaky_relu(val_mid, 0.1))
    # gnc.gat_run_add_to_center(at_gat, val_mid, att_mid, 128)
    # gnc.gat_run_div_each(at_gat, att_mid, val_mid, 128)

    # gnc.gcn_update_val(at, val_mid)
    # gnc.gcn_run(at, feat, output_feat, 128, 1)
    torch.cuda.synchronize()
    return output_feat

def gcn_layer_ours(feat, output_feat, weight):
    feat2 = torch.mm(feat, weight)
    gnc.gcn_run(at, feat2, output_feat, 128, 1)
    output_feat = F.relu(output_feat)
    torch.cuda.synchronize()
    return output_feat 


def gat_layer_ours(feat, output_feat, weight, weight_lr):
    feat2 = torch.mm(feat, weight)
    att_lr = torch.mm(feat2, weight_lr)
    #print(feat.shape)
    #print(output_feat.shape)
    #print(att_lr.shape)
    #print("---")
    gnc.gat_run(at_gat, feat2, att_lr, output_feat, 128, 1)
    torch.cuda.synchronize()
    return output_feat






def main(args):
    print("entry the main")
    num_heads = 1
    output_dim = 32 # 32 64 128 are fine
    model_type = args.model
    #print("model type={}".format(model_type))
    cudaid = args.gpu
    torch.cuda.set_device(cudaid)
    device = torch.device("cuda:{}".format(cudaid))
    #print("dset=" + str(args.syn_name))
    dset = args.syn_name
    #tdev = torch.cuda.get_device_name()
    #print(type(tdev))
    
    # load and preprocess dataset
    # data = load_data(args)
    num_v = 0
    num_e = 0
    with open("../data/{}.config".format(dset), 'r') as f:
        l = f.readline().split(' ')
        num_v = (int)(l[0])
        num_e = (int)(l[1])

    print("num_v", num_v, num_e)
    """
    h = torch.randn([num_v,512]).cuda()
    src_list = []
    dst_list = []
    t_load_begin = time.time()
    with open("../data/{}.graph".format(dset), 'r') as f:
        ptr = f.readline().strip("\n").strip(" ").split(" ")
        idx = f.readline().strip("\n").strip(" ").split(" ")
        for item in range(num_v):
            #print(item)
            which = (int)(item)
            selfloop = False
            #print("77", which)
            #print("88", ptr[which], ptr[which + 1])
            for i in range((int)(ptr[which]), (int)(ptr[which + 1])):
                dst_list.append(which)
                src_list.append((int)(idx[i]))
                if which == (int)(idx[i]):
                    selfloop = True
            # if not selfloop:
            #     dst_list.append(which)
            #     src_list.append(which)
    # print("load data time {}".format((str)(time.time() - t_load_begin)))


    #g = DGLGraph((src_list, dst_list)).to(device)
    # g = dgl.graph((src_list, dst_list)).to(device)
    """
    print("--------------------graph is load successfully----------------------")
    cuda = True

    #g.remove_edges_from(nx.selfloop_edges(g))
    # add self loop
    #if args.self_loop:
    #    g.remove_edges_from(nx.selfloop_edges(g))
    #    g.add_edges_from(zip(g.nodes(), g.nodes()))

    # g = DGLGraph(g)
    #n_edges = g.number_of_edges()
    #num_v = g.number_of_nodes()
    torch.manual_seed(123)

    vals = torch.ones(num_e).cuda()
    #ptrs, idxs = gnc.new_load(dset, "_thres_0.2", cudaid)
    #print("11", type(ptrs), type(idxs))
    #idxs = torch.load('/home/ygong07/exp/GraphPy_workflow/new_script/tensor_nebr.pt')
    #ptrs = torch.load('/home/ygong07/exp/GraphPy_workflow/new_script/tensor_off.pt')
    #idx_array = idxs.numpy()
    #int_array = idx_array.astype(int)
    #print("888", ptrs)
    #idxs = torch.from_numpy(int_array)
    #print(ptrs)
    #print(type(ptrs[0]))
    #print(idxs)
    #print(len(ptrs), ptrs.size())
    #print(type(idxs))
    #ptr_array = ptrs.numpy()
    #ptr_array = ptr_array.astype(int)
    #print("888", ptrs)
    #ptrs = torch.from_numpy(ptr_array)
    ptrs, idxs = gnc.new_load(dset, "", cudaid)
    print(idxs)
    at = gnc.gcn_init(ptrs, idxs, vals)
    gnc.gcn_schedule(at, 32)

    at_gat = gnc.gat_init(ptrs, idxs)
    gnc.gat_schedule(at_gat, 32)


    

    weight0 = torch.randn([512, output_dim]).cuda()
    weight1 = torch.randn([128, 64]).cuda()
    weight2 = torch.randn([64, 32]).cuda()
    #h = torch.randn([num_v, 512]).cuda()


    """
    def sagelstm_layer_dgl(feat, lstm):
        def _lstm_reducer(nodes):
            m = nodes.mailbox['m'] # (B, L, D)
            batch_size = m.shape[0]
            h = (m.new_zeros((1, batch_size, 32)), # 32 is the hidden feature size
                m.new_zeros((1, batch_size, 32))) # 32 is the hidden feature size
            tin0 = time.time()
            _, (rst, _) = lstm(m, h)
            print("in lstm {}".format(time.time() - tin0))
            return {'neigh': rst.squeeze(0)}
        g.srcdata['h'] = feat
        tt0 = time.time()
        g.update_all(fn.copy_src('h', 'm'), _lstm_reducer)
        torch.cuda.synchronize()
        print("lstm {}", time.time() - tt0)
        return g.dstdata['neigh']
    """
    runtimes = 1
    """
    if model_type == "GCN":
        def GCN_forward():
            feat1 = gcn_layer_dgl(h, weight0, 512, 128)
            feat2 = gcn_layer_dgl(feat1, weight1, 128, 64)
            feat3 = gcn_layer_dgl(feat2, weight2, 64, 32)

        # warmup
        for it in range(runtimes):
            GCN_forward()
        # run
        time0 = time.time()
        for it in range(runtimes):
            GCN_forward()
        print("DGL figure7 time {} {}".format(model_type, (time.time() - time0) / runtimes))
    elif model_type == "GAT":
        weight_l0 = torch.randn([128, 1]).cuda()
        weight_l1 = torch.randn([64, 1]).cuda()
        weight_l2 = torch.randn([32, 1]).cuda()

        weight_r0 = torch.randn([128, 1]).cuda()
        weight_r1 = torch.randn([64, 1]).cuda()
        weight_r2 = torch.randn([32, 1]).cuda()
        def GAT_forward():
            feat1 = gat_layer_dgl(h, weight0, weight_l0, weight_r0, 512, 128)
            feat2 = gat_layer_dgl(feat1, weight1, weight_l1, weight_r1, 128, 64)
            feat3 = gat_layer_dgl(feat2, weight2, weight_l2, weight_r2, 64, 32)

        # warmup
        for it in range(runtimes):
            GAT_forward()
        # run
        time0 = time.time()
        for it in range(runtimes):
            GAT_forward()
        print("DGL figure7 time {} {}".format(model_type, (time.time() - time0) / runtimes))
    """
    if model_type == "our_GCN":
        output_feat0 = torch.zeros([num_v, num_heads, 128]).cuda()
        output_feat1 = torch.zeros([num_v, 64]).cuda()
        output_feat2 = torch.zeros([num_v, 32]).cuda()

        def GCN_forward_ours():
            gcn_layer_ours(h, output_feat0, weight0)
            gcn_layer_ours(output_feat0, output_feat1, weight1)
            gcn_layer_ours(output_feat1, output_feat2, weight2)

        for it in range(runtimes):
            GCN_forward_ours()

        time0 = time.time()
        for it in range(runtimes):
            GCN_forward_ours()
        #print("our figure10 base time {} {}".format(model_type, (time.time() - time0) / runtimes))

    elif model_type == "our_GAT":
        print("-----------enter our GAT----------")
        # 2 is for both left and right part. (they connect them together)
        weight_lr0 = torch.randn([output_dim, 2]).cuda()
        weight_lr1 = torch.randn([64, 2]).cuda()
        weight_lr2 = torch.randn([32, 2]).cuda()

        output_feat0 = torch.zeros([num_v, num_heads, output_dim]).cuda()
        #output_feat1 = torch.zeros([num_v, 64]).cuda()
        #output_feat2 = torch.zeros([num_v, 32]).cuda()

        #def GAT_forward_ours():
        #    gat_layer_ours(h, output_feat0, weight0, weight_lr0)
        #    gat_layer_ours(output_feat0, output_feat1, weight1, weight_lr1)
        #    gat_layer_ours(output_feat1, output_feat2, weight2, weight_lr2)


        #for it in range(runtimes):
            #GAT_forward_ours()
            # gat_layer_our2(inp, output_feat, att_lr, att_mid, val_mid, 64, 32, at)
            #gat_layer_our(inp, output_feat, weight2, weight_lr, att_mid, val_mid, 64, 32, at)
            #gat_layer_our(h, output_feat0)

        time0 = time.time()
        for it in range(runtimes):
            #GAT_forward_ours()
            # gat_layer_our2(inp, output_feat, att_lr, att_mid, val_mid, 64, 32, at)
            #gat_layer_our(at_gat, at, h, output_feat0, weight0, weight_lr0, num_v, num_e, num_heads)
            test_spmmw_op(at, output_feat0, num_v, num_e, num_heads, output_dim)
        #print("our figure10 base time {} {}".format(model_type, (time.time() - time0) / runtimes))
    """
    elif model_type == "sagelstm":
        lstm = nn.LSTM(32, 32, batch_first=True).cuda()
        sage_input = torch.randn([num_v, 32]).cuda()
        for it in range(runtimes):
            sagelstm_layer_dgl(sage_input, lstm)
        time0 = time.time()
        for it in range(runtimes):
            sagelstm_layer_dgl(sage_input, lstm)
        print("DGL figure7 time {} {}".format(model_type, (time.time() - time0) / runtimes))

    
    elif model_type == "GAT_nograph":
        weight_l0 = torch.randn([128, 1]).cuda()
        weight_l1 = torch.randn([64, 1]).cuda()
        weight_l2 = torch.randn([32, 1]).cuda()

        weight_r0 = torch.randn([128, 1]).cuda()
        weight_r1 = torch.randn([64, 1]).cuda()
        weight_r2 = torch.randn([32, 1]).cuda()
        def GAT_forward_nograph():
            feat1 = gat_layer_dgl_no_graph(h, weight0, weight_l0, weight_r0, 512, 128)
            feat2 = gat_layer_dgl_no_graph(feat1, weight1, weight_l1, weight_r1, 128, 64)
            feat3 = gat_layer_dgl_no_graph(feat2, weight2, weight_l2, weight_r2, 64, 32)

        # warmup
        for it in range(runtimes):
            GAT_forward_nograph()
        # run
        time0 = time.time()
        for it in range(runtimes):
            GAT_forward_nograph()
        print("DGL figure7 time {} {}".format(model_type, (time.time() - time0) / runtimes))
    elif model_type == "GCN_nograph":
        def GCN_forward_nograph():
            feat1 = gcn_layer_dgl_no_graph(h, weight0, 512, 128)
            feat2 = gcn_layer_dgl_no_graph(feat1, weight1, 128, 64)
            feat3 = gcn_layer_dgl_no_graph(feat2, weight2, 64, 32)

        # warmup
        for it in range(runtimes):
            GCN_forward_nograph()
        # run
        time0 = time.time()
        for it in range(runtimes):
            GCN_forward_nograph()
        print("DGL figure7 time {} {}".format(model_type, (time.time() - time0) / runtimes))
    """




if __name__ == '__main__':
    print("1111at the begining_main---")
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--syn-name", type=str,
            help="")
    parser.add_argument("--model", type=str,
            help="", required=True)
    args = parser.parse_args()
    print(args)

    main(args)

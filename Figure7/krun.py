import argparse, time
import torch
import torch.nn as nn
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
    

def test_spmmw_op(at, output_feat, num_v, num_e, num_heads, output_dim, it):
    feat2 = torch.ones(num_v, output_dim).cuda();
    val_mid = torch.ones(num_e, num_heads).cuda()
    
    #feat1 = torch.load("/home/pkumar/src/GraphPy_workflow/new_script/data_save/spmmw_op2d_X.pt").cuda()
    #feat1 = torch.load('/home/ygong07/exp/GraphPy_workflow/new_script/reddit_spmmw_op2d_X.pt').cuda()
    
    #feat2 = feat1.view(num_v, output_dim) 
    #val_mid = torch.load("/home/pkumar/src/GraphPy_workflow/new_script/data_save/spmmw_op2d_Y.pt").cuda()
    
    #print("spmmw_op before:feat_output", output_feat)
    time0 = time.time()
    gnc.gcn_update_val(at, val_mid)
    torch.cuda.synchronize()
    gnc.gcn_run(at, feat2, output_feat, 128, 1)
    torch.cuda.synchronize()
    print("terms of " + str(it + 1) + "round our spmmw_op kernel time at round {}".format(time.time() - time0 ))
    print("size", val_mid.size(), feat2.size(), output_feat.size())
    print("spmmw_op after:feat_output", output_feat)
    #result = torch.load('/home/ygong07/exp/GraphPy_workflow/new_script/reddit_spmmw_op2d_result.pt').cuda()
    #print("spmmw2d results are the same", torch.all(output_feat.eq(result)))
    return output_feat


def main(args):
    print("entry the main")
    num_heads = 1
    output_dim = args.feat # 32 64 128 are fine
    cudaid = args.gpu
    torch.cuda.set_device(cudaid)
    device = torch.device("cuda:{}".format(cudaid))
    #print("dset=" + str(args.syn_name))
    dset = args.syn_name
    
    # load and preprocess dataset
    num_v = 0
    num_e = 0
    runtimes = 2
    with open(dset + "_graph_noeid.info", 'r') as f:
        l = f.readline().split('=')
        print(l)
        num_v = (int)(l[1])
        l = f.readline().split('=')
        print(l)
        num_e = (int)(l[1])

    print("num_v", num_v, num_e)
    cuda = True

    torch.manual_seed(123)

    vals = torch.ones(num_e).cuda()
    #ptrs, idxs = gnc.new_load(dset, "_thres_0.2", cudaid)
    ptrs, idxs = gnc.new_load(dset, "", cudaid)
    #print(idxs)
    at = gnc.gcn_init(ptrs, idxs, vals)
    gnc.gcn_schedule(at, 32)

    print("-----------enter our GAT----------")
    output_feat0 = torch.zeros([num_v, output_dim]).cuda()
    time0 = time.time()
    for it in range(runtimes):
        test_spmmw_op(at, output_feat0, num_v, num_e, num_heads, output_dim, it)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--syn-name", type=str,
            help="")
    parser.add_argument("--feat", type=int,
            help="", required=True)
    args = parser.parse_args()
    print(args)

    main(args)

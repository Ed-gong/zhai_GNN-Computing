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
    

def test_spmmv(at, output_feat, num_v, num_e, output_dim, it):
    feat2 = torch.ones(num_v, output_dim).cuda()
    time0 = time.time()
    gnc.gcn_run(at, feat2, output_feat, 128, 1)
    print("terms of " + str(it + 1) + "round our spmm kernel time at round {}".format(time.time() - time0 ))
    print("size", feat2.size(), output_feat.size())
    print("spmm after:feat_output", output_feat)
    torch.cuda.synchronize()
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
    for it in range(runtimes):
        test_spmmv(at, output_feat0, num_v, num_e, output_dim, it)


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

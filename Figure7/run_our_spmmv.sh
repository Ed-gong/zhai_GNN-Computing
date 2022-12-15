#dsets=(reddit.dgl)
prefix=/mnt/huge_26TB/data/
prefix2=results32_spmmv/
postfix=/saved_graph/
postfix2=_result_

dsets=(amazon as_skitter cit_patent com_dblp email_euall k21 ogb_product roadNet_CA soc_livejournal1 sx_stackoverflow web_berkstand wiki_talk ca_hollywood09 orkut reddit)

#dsets=(/mnt/huge_26TB/data/test2/reddit/saved_graph/)

for k in `seq 1 32`;
do
    #for j in `seq 0 4`;
    for i in `seq 0 14`;
    do
        #for i in `seq 0 14`;
        for j in `seq 0 4`;
        do
            #echo "CUDA_VISIBLE_DEVICES=0 python3 krun.py --syn-name ${prefix}${dsets[i]}${postfix}  --gpu 0 --feat ${k} >> ${prefix2}${dsets[i]}${postfix2}${k}".txt""
            CUDA_VISIBLE_DEVICES=0 python3 krun_spmmv.py --syn-name ${prefix}${dsets[i]}${postfix}  --gpu 0 --feat ${k} >> ${prefix2}${dsets[i]}${postfix2}${k}".txt"

            #CUDA_VISIBLE_DEVICES=0 python3 krun.py --syn-name ${prefix}${dsets[i]}${postfix}  --gpu 0 --feat ${k} >> ${prefix2}${dsets[i]}${postfix2}${k}".txt"
            #CUDA_VISIBLE_DEVICES=0 python3 krun.py --syn-name ${prefix}${dsets[i]}${postfix}  --gpu 0 --feat 32 >> ${prefix2}${dsets[i]}${postfix2}
            #CUDA_VISIBLE_DEVICES=0 python3 krun.py --syn-name  ${dsets[0]} --gpu 0 --feat 32
        done 
    done
done

#for i in `seq 0 1`;
#do
#    CUDA_VISIBLE_DEVICES=0 python3 our.py --syn-name  ${dsets[i]} --gpu 0 --model our_GCN  2>&1 | tee -a ./results/gcn_our.log
#done


#rm -f ./results/sage_our.log
#for i in `seq 0 1`;
#do
#    CUDA_VISIBLE_DEVICES=0 ./fig7.out --dataset ${dsets[i]}_sample_16 --feature-len 32 --nei 16 2>&1 | tee -a ./results/sage_our.log
#done


#cat ./results/gcn_our.log | grep figure | awk '{print $6}' > ./results/our_gcn_results.log
#cat ./results/gat_our.log | grep figure | awk '{print $6}' > ./results/our_gat_results.log
#cat ./results/sage_our.log | ack "timing_our|Cuda failure" | awk '{print $9}' > ./results/our_sage_results.log

prefix=/mnt/huge_26TB/data/
prefix2=results_com/
postfix=/saved_graph/
postfix2=_result_

dsets=(amazon as_skitter cit_patent com_dblp email_euall k21 ogb_product roadNet_CA soc_livejournal1 sx_stackoverflow web_berkstand wiki_talk ca_hollywood09 orkut reddit)


#for k in `seq 1 32`;
for k in 2 4 8 16 32;
do
    for j in `seq 0 4`;
    do
        #for i in `seq 0 14`;
        for i in 3;
        do
            #echo "CUDA_VISIBLE_DEVICES=0 python3 krun.py --syn-name ${prefix}${dsets[i]}${postfix}  --gpu 0 --feat ${k} >> ${prefix2}${dsets[i]}${postfix2}${k}".txt""
            echo "CUDA_VISIBLE_DEVICES=0 python3 krun.py --syn-name ${prefix}${dsets[i]}${postfix}  --gpu 0 --feat ${k} >> ${prefix2}${dsets[i]}${postfix2}${k}".txt""

            #CUDA_VISIBLE_DEVICES=0 python3 krun.py --syn-name ${prefix}${dsets[i]}${postfix}  --gpu 0 --feat ${k} >> ${prefix2}${dsets[i]}${postfix2}${k}".txt"
            #CUDA_VISIBLE_DEVICES=0 python3 krun.py --syn-name ${prefix}${dsets[i]}${postfix}  --gpu 0 --feat 32 >> ${prefix2}${dsets[i]}${postfix2}
            #CUDA_VISIBLE_DEVICES=0 python3 krun.py --syn-name  ${dsets[0]} --gpu 0 --feat 32
        done 
    done
done


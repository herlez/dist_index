#!/bin/bash -l
#SBATCH -N 12 --constraint=cstd01 --ntasks-per-node=20 --mem=60G -t 02:00:00 --output=/work/smarherl/rindex_results/results_%j.txt -p short

#cd /work/smarherl/ma/dist_index2/build/

#for F in /work/smarherl/matext/dna.64GiB/dna.64GiB
#do
#mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 1000000
#done

for F in /work/smarherl/matext/dna.4GiB/dna.4GiB /work/smarherl/matext/dewiki.4GiB/dewiki.4GiB
do
mpirun --mca osc ^rdma -n 20 ./bench 1 $F ${F}.q30 -q 1000000
done


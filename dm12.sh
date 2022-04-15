#!/bin/bash -l
#SBATCH -N 12 --constraint=cstd01 --ntasks-per-node=20 --mem=60G -t 02:00:00 --output=/work/smarherl/rindex_results/results_%j.txt -p short

cd /work/smarherl/ma/dist_index/build/

#strong scaling
for F in /work/smarherl/matext/dna.16GiB/dna.16GiB
do
#mpirun --mca osc ^rdma -n 960  ./bench 1 $F ${F}.q30 -q 1000000
#mpirun --mca osc ^rdma -n 720  ./bench 1 $F ${F}.q30 -q 1000000
#mpirun --mca osc ^rdma -n 480  ./bench 1 $F ${F}.q30 -q 1000000
#mpirun --mca osc ^rdma -n 360  ./bench 1 $F ${F}.q30 -q 1000000
mpirun --mca osc ^rdma -n 240  ./bench 1 $F ${F}.q30 -q 1000000
mpirun --mca osc ^rdma -n 160  ./bench 1 $F ${F}.q30 -q 1000000
mpirun --mca osc ^rdma -n 120  ./bench 1 $F ${F}.q30 -q 1000000
mpirun --mca osc ^rdma -n 90  ./bench 1 $F ${F}.q30 -q 1000000
mpirun --mca osc ^rdma -n 60  ./bench 1 $F ${F}.q30 -q 1000000
done

#weak scaling
for F in /work/smarherl/matext/dna.16GiB/dna.16GiB
do
mpirun --mca osc ^rdma -n 60 ./bench 1 $F ${F}.q30 -q 1000000
done

for F in /work/smarherl/matext/dna.32GiB/dna.32GiB
do
mpirun --mca osc ^rdma -n 120 ./bench 1 $F ${F}.q30 -q 1000000
done

for F in /work/smarherl/matext/dna.64GiB/dna.64GiB
do
mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 1000000
done

for F in /work/smarherl/matext/dna.64GiB/dna.64GiB
do
mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 1024
mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 2048
mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 4096
mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 8192
mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 16384
mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 32768
mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 65536
mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 131072
mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 262144
mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 524288
mpirun --mca osc ^rdma -n 240 ./bench 1 $F ${F}.q30 -q 1048576
done



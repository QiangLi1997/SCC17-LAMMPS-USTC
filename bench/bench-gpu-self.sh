#!/bin/sh
d=../../bin
mkdir results
for s in tersoff tersoff_gpu; do
  # GPU
  for p in single double mixed; do
    e=lmp_gpu_$p
    echo $e
    for i in `seq 0 5`; do
      $d/$e -in in.$s -v p vanilla -sf gpu > results/$s-$e-$i
      cat results/$s-$e-$i | grep -i Performance
    done
  done
  for v in vect novect; do
    e=lmp_kk_$v
    echo $e
    for i in `seq 0 5`; do
      $d/$e -in in.$s -v p kokkos -sf kk -k on t 0 g 1 > results/$s-$e-$i
      cat results/$s-$e-$i | grep -i Performance
    done
  done
done

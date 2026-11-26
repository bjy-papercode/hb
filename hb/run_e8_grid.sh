#!/bin/bash
# E8 + referee grid for the 2D experiments (GPU recommended; on a 2-core CPU one run of 6000 steps takes
# ~5 min for dsm/ssm and ~15-60 min with the heat-ball residual).
#   bash experiments/run_e8_grid.sh [main|lam|radius|nonlinear|learned|all] [N_PARALLEL]
# Outputs: results_e8/<tag>.json (+ _samples.npz); aggregate with
#   python experiments/aggregate.py results_e8 --window
set -e
cd "$(dirname "$0")/.."
PART=${1:-all}; NPAR=${2:-2}
SEEDS="0 1 2 3 4"
MAIN5="asymmetric_gmm sparse_dense_gmm elongated_gmm two_spiral ring"
SUPP3="double_ring pinwheel two_moon"
OUT=results_e8
jobs=()
add() { jobs+=("python3 -u experiments/train2d.py --out_dir $OUT $*"); }

if [[ $PART == main || $PART == all ]]; then
  # (R2.1-R2.3) same data, same network, same forward process; methods differ only in the objective.
  # linear forward processes (exact references; DSM allowed): isotropic OU and anisotropic OU + rotation
  for ds in $MAIN5 $SUPP3; do for dr in ou ou_aniso_rot; do for s in $SEEDS; do
    for m in dsm ssm; do add --dataset $ds --drift $dr --method $m --seed $s; done
    for m in dsm+hb dsm+fp ssm+hb; do add --dataset $ds --drift $dr --method $m --lam 0.1 --seed $s; done
  done; done; done
fi
if [[ $PART == lam || $PART == all ]]; then
  # (R1: gradient conflict / weighting) lambda ablation with gradient cosine
  for ds in asymmetric_gmm two_spiral; do for lam in 0.01 0.1 1; do for s in $SEEDS; do
    add --dataset $ds --drift ou --method dsm+hb --lam $lam --seed $s
    add --dataset $ds --drift ou --method dsm+fp --lam $lam --seed $s
  done; done; done
fi
if [[ $PART == radius || $PART == all ]]; then
  # (R1: sensitivity to r and tau_max) and the derivative-free (fd) variant
  for ds in asymmetric_gmm two_spiral; do for s in $SEEDS; do
    for r in 0.1 0.3 0.6; do add --dataset $ds --drift ou --method dsm+hb --lam 0.1 --r $r --seed $s; done
    for tm in 0.5 2.0; do add --dataset $ds --drift ou --method dsm+hb --lam 0.1 --tau_max $tm --seed $s; done
    add --dataset $ds --drift ou --method dsm+hb --lam 0.1 --hb_mode fd --seed $s
  done; done
fi
if [[ $PART == nonlinear || $PART == all ]]; then
  # (E8) fixed NONLINEAR forward drift (grid reference): ssm vs ssm+hb vs hb_only vs ssm+fp
  for ds in asymmetric_gmm two_spiral ring; do for s in $SEEDS; do
    add --dataset $ds --drift dw_rot --method ssm --seed $s
    add --dataset $ds --drift dw_rot --method ssm+hb --lam 0.1 --seed $s
    add --dataset $ds --drift dw_rot --method ssm+fp --lam 0.1 --seed $s
    add --dataset $ds --drift dw_rot --method hb_only --seed $s
  done; done
fi
if [[ $PART == learned ]]; then
  # learned drift from original_fixed/train_forward.py (frozen):  CKPT=path bash run_e8_grid.sh learned
  for s in $SEEDS; do for m in ssm ssm+hb ssm+fp; do
    add --dataset ${DS:-ring} --drift learned:$CKPT --method $m --lam 0.1 --seed $s
  done; done
fi
mkdir -p $OUT
printf '%s\n' "${jobs[@]}" > $OUT/joblist_$PART.txt
echo "${#jobs[@]} runs -> $OUT/joblist_$PART.txt ; running $NPAR in parallel"

cat $OUT/joblist_$PART.txt | xargs -P $NPAR -I{} bash -c '{} > /dev/null 2>&1 || echo "FAILED: {}"'
python3 experiments/aggregate.py $OUT --window

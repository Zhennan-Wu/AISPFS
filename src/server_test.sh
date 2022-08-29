declare -a ins_files
declare -a alg_choices


alg_choices=(random bwd_lbp fwd_lbp mfvi_bwd mfvi_fwd mfvi_bwd_noS csvi_bwd csvi_fwd exp_mfvi_bwd exp_csvi_bwd)

ins_files=(sysadmin_test1.spudd)


for ins in ${ins_files[@]}; do
    for alg in ${alg_choices[@]}; do
        job=${ins}_${alg}
        echo $problem $alg $job 
        sbatch -J $job --export=ALL,ALGORITHM=$alg,INS=$ins, pai.script
    done
done

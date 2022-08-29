declare -a ins_files
declare -a alg_choices


alg_choices=(random bwd_lbp fwd_lbp mfvi_bwd mfvi_fwd mfvi_bwd_noS csvi_bwd csvi_fwd exp_mfvi_bwd exp_csvi_bwd)


ins_files=(crossing_traffic_inst_mdp__1.spudd crossing_traffic_inst_mdp__2.spudd crossing_traffic_inst_mdp__3.spudd crossing_traffic_inst_mdp__4.spudd crossing_traffic_inst_mdp__5.spudd crossing_traffic_inst_mdp__6.spudd crossing_traffic_inst_mdp__7.spudd crossing_traffic_inst_mdp__8.spudd crossing_traffic_inst_mdp__9.spudd crossing_traffic_inst_mdp__10.spudd skill_teaching_inst_mdp__1.spudd skill_teaching_inst_mdp__2.spudd skill_teaching_inst_mdp__3.spudd skill_teaching_inst_mdp__4.spudd skill_teaching_inst_mdp__5.spudd skill_teaching_inst_mdp__6.spudd skill_teaching_inst_mdp__7.spudd skill_teaching_inst_mdp__8.spudd skill_teaching_inst_mdp__9.spudd skill_teaching_inst_mdp__10.spudd elevators_inst_mdp__1.spudd elevators_inst_mdp__2.spudd elevators_inst_mdp__3.spudd elevators_inst_mdp__4.spudd elevators_inst_mdp__5.spudd elevators_inst_mdp__6.spudd elevators_inst_mdp__7.spudd elevators_inst_mdp__8.spudd elevators_inst_mdp__9.spudd elevators_inst_mdp__10.spudd sysadmin_inst_mdp__1.spudd sysadmin_inst_mdp__2.spudd sysadmin_inst_mdp__3.spudd sysadmin_inst_mdp__4.spudd sysadmin_inst_mdp__5.spudd sysadmin_inst_mdp__6.spudd sysadmin_inst_mdp__7.spudd sysadmin_inst_mdp__8.spudd sysadmin_inst_mdp__9.spudd sysadmin_inst_mdp__10.spudd game_of_life_inst_mdp__1.spudd game_of_life_inst_mdp__2.spudd game_of_life_inst_mdp__3.spudd game_of_life_inst_mdp__4.spudd game_of_life_inst_mdp__5.spudd game_of_life_inst_mdp__6.spudd game_of_life_inst_mdp__7.spudd game_of_life_inst_mdp__8.spudd game_of_life_inst_mdp__9.spudd game_of_life_inst_mdp__10.spudd traffic_inst_mdp__1.spudd traffic_inst_mdp__2.spudd traffic_inst_mdp__3.spudd traffic_inst_mdp__4.spudd traffic_inst_mdp__5.spudd traffic_inst_mdp__6.spudd traffic_inst_mdp__7.spudd traffic_inst_mdp__8.spudd traffic_inst_mdp__9.spudd traffic_inst_mdp__10.spudd)


for ins in ${ins_files[@]}; do
    for alg in ${alg_choices[@]}; do
        job=${alg}_${ins:0:4}_${ins:-1}
        echo $problem $alg $job 
        sbatch -J $job --export=ALL,ALGORITHM=$alg,INS=$ins, pai.script
    done
done

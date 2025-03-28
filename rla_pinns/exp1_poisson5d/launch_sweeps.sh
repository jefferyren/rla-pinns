# Launch all sweeps using the sbatch command
cd sweeps/

# Benchamrks
# sbatch SGD.sh
# sbatch Adam.sh
# sbatch ENGD.sh
# sbatch SPRING.sh

# Woodbury ENGD
sbatch ENGD_woodbury_exact.sh
# sbatch ENGD_woodbury_naive_10.sh
# sbatch ENGD_woodbury_naive_50.sh
# sbatch ENGD_woodbury_nystrom_10.sh
# sbatch ENGD_woodbury_nystrom_50.sh

# Spring
sbatch SPRING_exact.sh
# sbatch SPRING_naive_10.sh
# sbatch SPRING_naive_50.sh
# sbatch SPRING_nystrom_10.sh
# sbatch SPRING_nystrom_50.sh

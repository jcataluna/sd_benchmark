# venv
python -m venv tpuenv
source tpuenv/bin/activate

# jax-tpu
pip install "jax[tpu]==0.4.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -c "import jax; print(jax.device_count()); print(jax.local_device_count())"

# sd on tpus
git clone https://github.com/jcataluna/sd_benchmark.git
cd sd_benchmark/
pip install -r requirements.txt 
python sd_tpu_one_run.py 


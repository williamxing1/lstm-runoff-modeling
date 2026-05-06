import modal

app = modal.App("lstm-runoff-modeling-inference")
image = (
    modal.Image.debian_slim()
    .pip_install("pandas", "numpy", "torch", "matplotlib")
    .add_local_file("../data_code/dataset.py", "/dataset.py")
    .add_local_file("../models/lstm.py", "/lstm.py")
    .add_local_file("../models/transformer.py", "/transformer.py")
    .add_local_file("nse_cdf.py", "/nse_cdf.py")
    .add_local_file("../training/train.py", "/train.py")
)

data_volume = modal.Volume.from_name("lstm-runoff-modeling-data")
weights_volume = modal.Volume.from_name("lstm-runoff-modeling-outputs")
outputs_volume = modal.Volume.from_name("lstm-runoff-modeling-inference-outputs")

@app.function(
    image=image,
    gpu="A100",
    timeout=60*60,
    volumes={
        "/data": data_volume,
        "/weights": weights_volume,
        "/outputs": outputs_volume
    }
)
def train():
    import subprocess
    subprocess.run(["python3", "-u", "/nse_cdf.py"], check=True)

@app.local_entrypoint()
def main():
    train.remote()
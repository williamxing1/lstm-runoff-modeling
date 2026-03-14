import modal

app = modal.App("lstm-runoff-modeling")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "pandas", "numpy", "matplotlib")
    .add_local_file("dataset_all.py", "/dataset_all.py")
    .add_local_file("lstm.py", "/lstm.py")
    .add_local_file("transformer.py", "/transformer.py")
    .add_local_file("train.py", "/train.py")
)

data_volume = modal.Volume.from_name("lstm-runoff-modeling-data")
outputs_volume = modal.Volume.from_name("lstm-runoff-modeling-outputs")

@app.function(
    image=image,
    gpu="A100",
    timeout=60*60,
    volumes={
        "/data": data_volume,
        "/outputs": outputs_volume
    }
)
def train():
    import subprocess
    subprocess.run(["python3", "-u", "/train.py"], check=True)

@app.local_entrypoint()
def main():
    train.remote()
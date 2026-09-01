import json
import argparse

from comet import download_model, load_from_checkpoint


def get_comet_model():
    """Retrieves a comet model.

    Returns:
        Cached comet model.
    """
    model_path = "/reward_models/wmt22-cometkiwi-da/checkpoints/model.ckpt"

    model = load_from_checkpoint(model_path, reload_hparams=True, local_files_only=True)
    return model


def comet_score(model, data):
    samples = [
        {
            "src": example["english"],
            "mt": example["chosen"],
        }
        for example in data
    ]

    model_output = model.predict(
        samples=samples,
        batch_size=8,
        gpus=1,
        num_workers=0,
        accelerator="cuda"
    )

    return model_output[0]


def main(args):
    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]

    model = get_comet_model()
    scores = comet_score(model, data)

    with open(args.output_path, "w") as f:
        for example, score in zip(data, scores):
            example["comet_score"] = score
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(args)

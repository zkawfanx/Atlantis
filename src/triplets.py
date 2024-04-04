import os

import datasets
import pandas as pd

_VERSION = datasets.Version("0.0.1")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "text": datasets.Value("string"),
        "image": datasets.Image(),
        "conditioning_image": datasets.Image(),
    },
)

METADATA_PATH = "/path/to/metadata.jsonl"
IMAGES_DIR = "/path/to/underwater/folder"
CONDITIONING_IMAGES_DIR = "/path/to/depth/folder"

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)

class Depth2Underwater(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_path = dl_manager.download(METADATA_PATH)
        images_dir = dl_manager.download(IMAGES_DIR)
        conditioning_images_dir = dl_manager.download(CONDITIONING_IMAGES_DIR)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                    "conditioning_images_dir": conditioning_images_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, conditioning_images_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            text = row["text"]

            image_path = row["image"]
            image_path = os.path.join(images_dir, image_path)
            image = open(image_path, "rb").read()

            conditioning_image_path = row["conditioning_image"]
            conditioning_image_path = os.path.join(
                conditioning_images_dir, row["conditioning_image"]
            )
            conditioning_image = open(conditioning_image_path, "rb").read()

            yield row["image"], {
                "text": text,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "conditioning_image": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
            }
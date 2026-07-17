"""One-time conversion of the legacy TF1 frozen graph to ONNX.

The 2021 project shipped its trained DenseNet-121 as a TensorFlow 1.x
frozen GraphDef (``frozen_graph.pb``). This script converts it — trained
weights included — to ONNX so the modern app can serve it with PyTorch
(via onnx2torch) without any TensorFlow dependency.

Usage (needs tensorflow + tf2onnx, see comments below — the serving app
itself does NOT need TensorFlow):

    pip install "tensorflow==2.16.2" tf2onnx "numpy<2"
    python scripts/convert_model.py path/to/frozen_graph.pb

Graph interface (from the original freeze_graph.py):
    inputs — float32 [N, 224, 224, 3], RGB scaled to [0, 1]
    densenet121/densenet121/TargetSpatialSqueeze — raw logits [N, 2]
    densenet121/probability — softmax over [normal, abnormal]

The original graph also had a ``labels`` input feeding
``masked_logits = logits * one_hot(labels)``, used only to pick which
class to differentiate for SmoothGrad. Exporting the raw logits instead
is equivalent (select the class column in PyTorch) and avoids the ONNX
``OneHot`` op, which onnx2torch does not implement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tf2onnx

DEFAULT_INPUT = Path(__file__).resolve().parent.parent.parent / "xray-vision-master" / "frozen_graph.pb"
OUTPUT = Path(__file__).resolve().parent.parent / "model" / "densenet121_mura.onnx"


def main() -> None:
    frozen_graph = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT

    graph_def = tf2onnx.tf_loader.tf.compat.v1.GraphDef()
    graph_def.ParseFromString(frozen_graph.read_bytes())

    model_proto, _ = tf2onnx.convert.from_graph_def(
        graph_def,
        input_names=["inputs:0"],
        output_names=[
            "densenet121/densenet121/TargetSpatialSqueeze:0",  # raw logits
            "densenet121/probability:0",
        ],
        opset=13,
        output_path=str(OUTPUT),
    )
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")
    print("Inputs :", [i.name for i in model_proto.graph.input])
    print("Outputs:", [o.name for o in model_proto.graph.output])


if __name__ == "__main__":
    main()

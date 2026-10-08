# Speaker diarization model attribution

Sidekick's speaker labels use **pyannote speaker-diarization-community-1**
(https://huggingface.co/pyannote/speaker-diarization-community-1, revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`)
by Hervé Bredin, Alexis Plaquet and the pyannote / pyannoteAI contributors, released under the
Creative Commons Attribution 4.0 International license (CC-BY-4.0,
https://creativecommons.org/licenses/by/4.0/).

The pipeline bundles:

- the pyannote powerset speaker segmentation model (Plaquet & Bredin, *Powerset multi-class cross
  entropy loss for neural speaker diarization*, Interspeech 2023);
- the WeSpeaker ResNet34 speaker embedding model trained on VoxCeleb
  (Wang et al., *Wespeaker: A research and production oriented speaker embedding learning toolkit*,
  ICASSP 2023; https://github.com/wenet-e2e/wespeaker, copied by pyannote from
  https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM). It follows the VoxCeleb dataset
  license, CC-BY-4.0 (https://mm.kaist.ac.kr/datasets/voxceleb/);
- the VBx PLDA model trained by the BUT Speech@FIT group (Landini et al., *Bayesian HMM clustering of
  x-vector sequences (VBx) in speaker diarization*, Computer Speech & Language 2022).

The clustering code Sidekick reimplements comes from pyannote.audio 4.x (MIT license,
https://github.com/pyannote/pyannote-audio).

## Changes made by Sidekick

These files are not the original checkpoints. Sidekick converted them:

- `community-1-segmentation.onnx`: the segmentation model exported to ONNX (opset 17), unchanged weights.
- `community-1-embedding.onnx`: the ResNet34 embedding network exported to ONNX (opset 17), unchanged weights.
  Kaldi filterbank extraction moved out of the graph into Sidekick's Rust code, and the
  segmentation-frame weights are resampled to the network's frame rate before the graph runs.
- `community-1-plda.json`: the PLDA (`plda/xvec_transform.npz`, `plda/plda.npz`) pre-solved into the whitened
  transform pyannote.audio derives at load time, saved as JSON alongside the community-1
  `config.yaml` clustering hyperparameters.

Converted to ONNX by Sidekick (https://github.com/royce-mathew/sidekick). No endorsement by the
original authors is implied.

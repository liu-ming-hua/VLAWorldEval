# License scope

The project owner selected MIT on 2026-10-12 for original VLAWorldEval framework
contributions. See [LICENSE](LICENSE). This grant covers only contributions that
VLAWorldEval contributors are entitled to license; it does not relicense copied
upstream code, model weights, datasets, or separately installed dependencies.

## Third-party boundaries

- WorldGym network code is obtained separately by
  `scripts/setup/worldgym_runtime.sh`, into `third_party/world-model-eval` and
  `third_party/worldgym_runtime`. It is excluded from source and wheel packages.
  Previous copies remain in `third_party/legacy_networks/worldgym`, also excluded.
  The inspected upstream Git
  tree at `709665b29f5c95124e7dbc05b0d4d47cd353de97` has no root license file.
  Its redistribution permission remains unconfirmed; these files are not
  granted MIT by this project's LICENSE. Their embedded references to GLIDE
  and Diffusion Forcing also require retention/review of upstream terms.
- `third_party/legacy_networks/rt1/` preserves unused implementations with references to Google's
  Open X-Embodiment and Robotics Transformer. Preserve upstream attribution
  and applicable terms; do not treat these implementations as original MIT
  framework code merely because they reside in this workspace. They are not
  packaged; the active adapter loads the separately downloaded TF SavedModel.
- Other upstream clones in `third_party/` retain their own terms and are
  excluded from source and wheel packages. Environments, weights, private
  data, caches and generated results are also excluded.
- Optional trajectory evaluation uses a separately downloaded EWMBench detector
  and Ultralytics runtime. Their upstream terms are not replaced by MIT; the
  detector model card states CC BY-NC-SA4.0. The inspected official source is
  under excluded `third_party/ewmbench/`; its metric implementation is not copied
  into the framework. Standard trajectory distances are independently implemented
  with explicit protocol differences, not advertised as identical official scores.

Using a model, redistributing its weights, and redistributing its implementation
are different boundaries. Contacting WorldGym authors is not a prerequisite
imposed by this framework for local inference, and no message has been sent.
For public distribution of inherited code, confirm an applicable upstream grant
or review a distribution that obtains the runtime separately. Neither route
changes model/data terms or implies unrestricted permission for all uses.

Selected demo media under `docs/assets/` are derived presentation assets, not a
relicensing of the underlying data or models. Their provenance is recorded there.
See [release boundaries](docs/en/release.md) for details.

"""Native20k-EMA scoring view; same checkpoint/runtime, no invented border pixels."""
from adapters.base import WorldModelAdapter
from adapters.world_models.cosmos3_nano_droid_fd import Cosmos3NanoDroidFDAdapter
from evaluation.contracts import WorldModelOutput


class Cosmos3DroidFDNativeAdapter(Cosmos3NanoDroidFDAdapter):
    """Keep the decoder's actual output instead of the policy bridge's edge fill.

    The full initial image still conditions inference. The preparation layer
    supplies an identically cropped display anchor and recorded-future region.
    This class is not a new checkpoint or a replacement for policy adapters.
    """

    supports_episode_reuse = True
    supports_native_batch = True

    def native_batch_step(self, items):
        """Independent trajectories, one recurrent step each; never mix history."""
        import json
        from pathlib import Path
        import av
        import numpy as np
        import torch.distributed as dist
        if not items or len(items) > self.inference_batch_size or self.diffusion_cache:
            raise ValueError("Invalid batch size or cross-sample diffusion cache enabled")
        rank = dist.get_rank() if dist.is_initialized() else 0
        samples = []
        for item in items:
            image = self._to_image(item["image"])
            actions = self.validate_action(item["action"])
            directory = Path(item["runtime_dir"]).resolve()
            if rank == 0:
                directory.mkdir(parents=True, exist_ok=True)
                image.save(directory/"condition.png")
                (directory/"actions.json").write_text(json.dumps(actions.tolist()))
            if dist.is_initialized():
                dist.barrier()
            sample = self._sample_overrides_cls(
                name=directory.name, model_mode="forward_dynamics",
                vision_path=str(directory/"condition.png"), action_path=str(directory/"actions.json"),
                domain_name="droid_lerobot", action_chunk_size=16, image_size=480,
                aspect_ratio="4,3", resolution=self.resolution, fps=self.fps,
                num_frames=17, num_steps=self.num_inference_steps, guidance=self.guidance,
                shift=self.shift, prompt="", seed=int(item["seed"]))
            sample.output_dir = directory/"outputs"
            sample.download(sample.output_dir/"inputs")
            samples.append(sample.build_sample(model_config=self._pipe.model_config))
        outputs = self._pipe.generate(samples)
        # Only rank zero transports videos. All ranks participate in generation.
        if rank != 0:
            return None
        if len(outputs) != len(items):
            raise RuntimeError("Batch output count differs from independent inputs")
        results = []
        for item, output in zip(items, outputs):
            if output.status != "success" or not output.outputs:
                raise RuntimeError(f"Native batched generation failed: {output}")
            path = next(p for p in output.outputs[0].files if Path(p).suffix == ".mp4")
            with av.open(str(path)) as container:
                frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
            if len(frames) != 17:
                raise ValueError(f"Expected 17 native frames, got {frames.shape}")
            video = frames[1:]
            results.append({"views": {self.view_names[0]: video}, "robot_state": None,
                "metadata": {"native_evaluation": True, "edge_filled_pixels": False,
                    "batch_size": len(items), "seed": item["seed"],
                    "pixel_transport": "decoded_upstream_mp4_before_framework_video_encoding"}})
        return results

    def set_episode_context(self, runtime_dir):
        """Keep weights resident; isolate artifacts before seed/reset RPCs."""
        from pathlib import Path
        import json
        if self.diffusion_cache:
            raise ValueError("Cross-episode native reuse requires diffusion_cache=false")
        config = json.loads((self.model_path/"config.json").read_text())
        if config["model"]["config"]["tokenizer"].get("keep_decoder_cache", False):
            raise ValueError("Cross-episode native reuse requires per-call decoder cache clearing")
        self.runtime_dir = Path(runtime_dir).resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def step_output(self, action, *, output_views=(), state_schema=None):
        conditioning_wh = list(self.current_image.size)
        result = WorldModelAdapter.step_output(self, action, output_views=output_views, state_schema=state_schema)
        return WorldModelOutput(views=result.views, robot_state=result.robot_state,
            metadata={**result.metadata, "native_evaluation": True,
                "conditioning_image_wh": conditioning_wh, "edge_filled_pixels": False,
                "pixel_transport": "decoded_upstream_mp4_before_framework_video_encoding",
                "decoded_hwc": {name: list(video.shape[1:]) for name, video in result.views.items()}})

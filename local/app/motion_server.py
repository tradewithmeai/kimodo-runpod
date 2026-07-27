"""
Text-to-motion server: MDM sampling -> 22-joint skeleton JSON -> browser viewer.

Deliberately bypasses MDM's dataset loader. sample/generate.py calls
get_dataset_loader(split='test'), which requires the full HumanML3D dataset (a large,
Drive-hosted download) purely to obtain the mean/std normalisation vectors and a
num_actions attribute. Both are avoidable:
  - t2m_mean.npy / t2m_std.npy ship inside the repo under dataset/
  - get_model_args() only does hasattr(data.dataset, 'num_actions'), so a stub suffices
Skipping the dataset also avoids SMPL mesh assets entirely, since we render the
joint positions recovered by recover_from_ric rather than a body mesh.

Run on the pod:
    /workspace/envs/mdm/bin/python motion_server.py
Serves on :8888, which RunPod already proxies.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

REPO = Path(os.environ.get('MDM_REPO', '/workspace/repos/motion-diffusion-model'))
# Resolve before chdir — MDM resolves some asset paths relative to its own repo root,
# so we move there, which would otherwise strand a relative path to viewer.html.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from utils.model_util import create_model_and_diffusion, load_model_wo_clip  # noqa: E402
from data_loaders.humanml.scripts.motion_process import recover_from_ric  # noqa: E402
from model.cfg_sampler import ClassifierFreeSampleModel  # noqa: E402
import model.mdm as _mdm  # noqa: E402


class _NoRotation2xyz:
    """
    MDM.__init__ unconditionally constructs Rotation2xyz, which loads
    body_models/smpl/SMPL_NEUTRAL.pkl. That pickle holds chumpy arrays, and chumpy is
    uninstallable on Python 3.12 (its setup.py imports pip). We never call rot2xyz:
    for the humanml dataset the network emits the 263-dim HML vector and joint
    positions come from recover_from_ric, so the SMPL mesh path is dead weight.
    Stubbing it avoids the dependency entirely.
    """

    class _NoSMPL:
        """MDM._apply and MDM.train forward into rot2xyz.smpl_model; absorb those."""

        def _apply(self, fn):
            return self

        def train(self, mode=True):
            return self

        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

    def __init__(self, *args, **kwargs):
        self.smpl_model = self._NoSMPL()

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            'rot2xyz is stubbed out; this server recovers joints via recover_from_ric.'
        )


_mdm.Rotation2xyz = _NoRotation2xyz

FPS = 20          # HumanML3D is 20fps
N_JOINTS = 22
MAX_FRAMES = 196  # MDM's training horizon; longer requests get clamped

# HumanML3D / SMPL 22-joint kinematic chains, used for drawing bones in the viewer.
KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 11],       # right leg
    [0, 1, 4, 7, 10],       # left leg
    [0, 3, 6, 9, 12, 15],   # spine -> head
    [9, 14, 17, 19, 21],    # right arm
    [9, 13, 16, 18, 20],    # left arm
]

JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
    'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
]


class _StubDataset:
    """create_model_and_diffusion only probes hasattr(data.dataset, 'num_actions')."""


class _StubData:
    dataset = _StubDataset()


def load_args(model_path: Path) -> argparse.Namespace:
    """MDM writes args.json beside the checkpoint; replay it as the model config."""
    args_path = model_path.parent / 'args.json'
    with open(args_path) as f:
        cfg = json.load(f)
    args = argparse.Namespace(**cfg)
    # Fields newer code reads that old checkpoints predate.
    for key, default in [
        ('pred_len', 0), ('context_len', 0), ('emb_policy', 'add'),
        ('multi_target_cond', False), ('multi_encoder_type', 'multi'),
        ('target_enc_layers', 1), ('lambda_vel', 0.0), ('lambda_rcxyz', 0.0),
        ('lambda_fc', 0.0),
    ]:
        if not hasattr(args, key):
            setattr(args, key, default)
    return args


class MotionModel:
    def __init__(self, model_path: Path, device: str = 'cuda'):
        self.device = device
        self.args = load_args(model_path)
        print(f'[init] dataset={self.args.dataset} steps={self.args.diffusion_steps} '
              f'arch={getattr(self.args, "arch", "?")}', flush=True)

        self.model, self.diffusion = create_model_and_diffusion(self.args, _StubData())
        # MDM checkpoints are plain state dicts, so the safe loader should work. This
        # weight file comes from a third-party Drive link, and weights_only=False would
        # unpickle arbitrary objects from it — only fall back if genuinely necessary.
        try:
            state = torch.load(model_path, map_location='cpu', weights_only=True)
        except Exception as exc:
            print(f'[warn] safe load failed ({exc.__class__.__name__}); checkpoint '
                  f'contains non-tensor objects. Falling back to unsafe load.', flush=True)
            state = torch.load(model_path, map_location='cpu', weights_only=False)
        load_model_wo_clip(self.model, state)
        # MDM overrides _apply without returning, so nn.Module.to() yields None here.
        # Upstream generate.py never chains these calls either — don't chain.
        self.model.to(device)
        self.model.eval()

        # Normalisation stats that inv_transform would otherwise pull off the dataset.
        self.mean = np.load(REPO / 'dataset' / 't2m_mean.npy')
        self.std = np.load(REPO / 'dataset' / 't2m_std.npy')

        self._cfg_model = ClassifierFreeSampleModel(self.model)
        self._cfg_model.to(device)
        self._cfg_model.eval()

    @torch.no_grad()
    def generate(self, text: str, seconds: float = 6.0, guidance: float = 2.5,
                 seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        n_frames = int(min(max(seconds, 1.0) * FPS, MAX_FRAMES))
        bs = 1

        model = self._cfg_model if guidance != 1.0 else self.model
        model_kwargs = {'y': {
            'mask': torch.ones(bs, 1, 1, n_frames, dtype=torch.bool, device=self.device),
            'lengths': torch.tensor([n_frames] * bs, device=self.device),
            'text': [text],
            'scale': torch.ones(bs, device=self.device) * guidance,
        }}

        shape = (bs, self.model.njoints, self.model.nfeats, n_frames)
        sample = self.diffusion.p_sample_loop(
            model, shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # (bs, 263, 1, frames) -> (bs, 1, frames, 263), de-normalise, then to joint xyz.
        x = sample.cpu().permute(0, 2, 3, 1).float().numpy()
        x = x * self.std + self.mean
        joints = recover_from_ric(torch.from_numpy(x).float(), N_JOINTS)
        joints = joints[0, 0].numpy()  # (frames, 22, 3)

        # Drop the skeleton onto the ground plane so the viewer doesn't have to guess.
        joints[:, :, 1] -= joints[:, :, 1].min()
        return joints


app = FastAPI()
MODEL: MotionModel | None = None
VIEWER = HERE / 'viewer.html'


class GenerateRequest(BaseModel):
    text: str
    seconds: float = 6.0
    guidance: float = 2.5
    seed: int | None = None


@app.get('/', response_class=HTMLResponse)
def index():
    return VIEWER.read_text(encoding='utf-8')


@app.get('/health')
def health():
    return {
        'ready': MODEL is not None,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'steps': MODEL.args.diffusion_steps if MODEL else None,
    }


@app.post('/generate')
def generate(req: GenerateRequest):
    if MODEL is None:
        return JSONResponse({'error': 'model not loaded'}, status_code=503)
    import time
    t0 = time.time()
    joints = MODEL.generate(req.text, req.seconds, req.guidance, req.seed)
    elapsed = time.time() - t0
    print(f'[gen] {elapsed:.1f}s  "{req.text}"  {joints.shape[0]} frames', flush=True)
    return {
        'text': req.text,
        'fps': FPS,
        'frames': int(joints.shape[0]),
        'joint_names': JOINT_NAMES,
        'chains': KINEMATIC_CHAINS,
        'elapsed_sec': round(elapsed, 2),
        # rounded to keep the payload small; sub-mm precision is irrelevant for display
        'joints': np.round(joints, 4).tolist(),
    }


if __name__ == '__main__':
    import uvicorn
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        REPO / 'save' / 'humanml_enc_512_50steps' / 'model000750000.pt'
    MODEL = MotionModel(ckpt)
    print('[init] model ready', flush=True)
    uvicorn.run(app, host='0.0.0.0', port=8888, log_level='warning')

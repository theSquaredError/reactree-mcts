import logging
from datetime import datetime

from omegaconf import OmegaConf


log = logging.getLogger(__name__)


class WandbLogger:
    def __init__(self, cfg, phase: str):
        self.enabled = False
        self.run = None

        enable_wandb = bool(OmegaConf.select(cfg, "wandb.enable", default=False))
        if not enable_wandb:
            return

        try:
            import wandb  # type: ignore
        except ImportError:
            log.warning("wandb is not installed. Run `pip install wandb` to enable online logging.")
            return

        project = OmegaConf.select(cfg, "wandb.project", default="reactree")
        entity = OmegaConf.select(cfg, "wandb.entity", default=None)
        mode = OmegaConf.select(cfg, "wandb.mode", default="online")
        group = OmegaConf.select(cfg, "wandb.group", default=None)
        tags = OmegaConf.select(cfg, "wandb.tags", default=[])

        run_name = OmegaConf.select(cfg, "wandb.name", default="")
        if not run_name:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"{cfg.name}-{phase}-{ts}"

        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                group=group,
                tags=tags,
                mode=mode,
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=cfg.out_dir,
                reinit=True,
            )
            self.enabled = self.run is not None
            if self.enabled:
                log.info("W&B logging enabled: project=%s, run=%s", project, run_name)
        except Exception as e:
            log.warning("Failed to initialize wandb. Continuing without online logging. Error: %s", repr(e))
            self.enabled = False
            self.run = None

    def log(self, metrics: dict, step: int = None):
        if not self.enabled:
            return
        if step is None:
            self.run.log(metrics)
        else:
            self.run.log(metrics, step=step)

    def finish(self):
        if self.enabled and self.run is not None:
            self.run.finish()
